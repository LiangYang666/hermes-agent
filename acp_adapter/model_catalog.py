"""ACP model picker: deduplicated ``provider:model`` rows from the Hermes inventory + named endpoints."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from acp.schema import ModelInfo, SessionModelState

logger = logging.getLogger("acp_adapter.server")

# Per-provider row cap (clients render all `availableModels` in one dropdown; mirrors the
# MoA picker cap). Not a total cap; the current model is always kept via the fallback insert.
ACP_MAX_MODELS_PER_PROVIDER = 200


def _named_custom_provider_catalogs() -> list[tuple[str, str, list[tuple[str, str]]]]:
    """``(slug, label, [(model_id, description), ...])`` for named endpoints (v12 ``providers:``
    and legacy ``custom_providers:``), which canonical provider enumeration never lists.

    Models = the entry's declared models, refreshed from the live ``/models`` listing when a
    credential exists and ``discover_models`` isn't disabled; declared models survive a failed
    discovery (some endpoints have no ``/models`` route). Slugs use the ``custom:<name>`` shape
    ``parse_model_input``/``resolve_runtime_provider`` resolve, so choice ids round-trip."""
    try:
        from hermes_cli.config import (get_compatible_custom_providers, is_provider_enabled, load_config)
        from hermes_cli.model_switch import _declared_model_ids, _entry_models_discovered, _models_config_is_allowlist
        from hermes_cli.model_switch_providers import _NativePickerModelList, _fetch_picker_live_models
        from hermes_cli.model_switch_providers import _discover_flag
        from hermes_cli.models_local import should_use_ollama_native_catalog
        from hermes_cli.providers import custom_provider_slug
    except ImportError:
        return []

    try:
        cfg = load_config()
        entries = get_compatible_custom_providers(cfg)
    except Exception:
        logger.debug("Could not load named custom providers", exc_info=True)
        return []

    # ``get_compatible_custom_providers`` drops ``enabled``; read disabled keys from raw config.
    raw_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    disabled_keys = {
        str(key).strip().lower()
        for key, raw in (raw_providers.items() if isinstance(raw_providers, dict) else ())
        if isinstance(raw, dict) and not is_provider_enabled(raw)
    }

    def _entry_catalog(entry: dict) -> tuple[str, str, list[tuple[str, str]]] | None:
        field = lambda key: str(entry.get(key) or "").strip()
        provider_key, name, base_url = field("provider_key"), field("name"), field("base_url")
        if provider_key.lower() in disabled_keys or not name or not base_url:
            return None
        slug = custom_provider_slug(name, provider_key)

        api_key = field("api_key")
        if not api_key:
            key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
            api_key = os.environ.get(key_env, "").strip() if key_env else ""

        models_cfg = entry.get("models")
        declared = [m for m in dict.fromkeys([field("model"), *_declared_model_ids(models_cfg)]) if m]

        native_headers = entry.get("extra_headers") or None
        is_ollama_key = provider_key.lower() in {"ollama", "custom:ollama"}
        is_native_ollama = should_use_ollama_native_catalog(
            provider_key if is_ollama_key else "custom", base_url, headers=native_headers
        )
        if not api_key and not declared and not is_native_ollama:
            return None  # nothing to discover with and nothing declared: not addressable

        model_ids = list(declared)
        live = None
        if _discover_flag(entry) and (api_key or is_native_ollama):
            try:
                live = _fetch_picker_live_models(
                    api_key, base_url, provider_key if is_native_ollama and is_ollama_key else "custom",
                    _models_config_is_allowlist(models_cfg, _entry_models_discovered(entry)),
                    headers=native_headers, timeout=1.5, api_mode=entry.get("api_mode"),
                )
            except Exception:
                live = None
            if isinstance(live, _NativePickerModelList):
                model_ids = list(live)
            elif live is not None:
                model_ids = declared + [m for m in live if m not in declared]

        if not model_ids and not isinstance(live, _NativePickerModelList):
            return None
        return slug, name, [(mid, "") for mid in model_ids]

    catalogs = [_entry_catalog(entry) for entry in entries if isinstance(entry, dict)]
    return [c for c in catalogs if c is not None]


def _semantic_provider(provider_id: str, normalize_provider: Callable[[str], str]) -> str:
    raw = str(provider_id or "").strip().lower()
    if raw in {"ollama", "custom:ollama"}:
        return "ollama"
    if raw.startswith("custom:"):
        return raw
    return normalize_provider(raw)


def _empty_catalog_applies(
    provider_id: str, empty_authoritative: set[str], normalize_provider: Callable[[str], str]
) -> bool:
    """True when a named endpoint with an authoritative-empty catalog owns ``provider_id``."""
    raw = str(provider_id or "").strip().lower()
    normalized = normalize_provider(raw)
    if normalized == "custom":
        return any(
            candidate == raw
            or f"custom:{candidate}" == raw
            or (raw == "custom" and candidate == "custom")
            for candidate in empty_authoritative
        )
    return any(
        candidate == raw
        or candidate == f"custom:{normalized}"
        or candidate == f"custom:{raw}"
        or normalize_provider(candidate) == normalized
        for candidate in empty_authoritative
    )


def _choice_provider(model_id: str) -> str:
    """Provider prefix of an encoded choice id; longest configured ``custom:`` slug wins."""
    parts = model_id.split(":")
    if parts[:1] == ["custom"] and len(parts) > 1:
        from hermes_cli.models import _configured_custom_provider_ids

        lowered = model_id.lower()
        for candidate in sorted(
            (p for p in _configured_custom_provider_ids() if p.startswith("custom:")), key=len, reverse=True,
        ):
            if lowered.startswith(candidate + ":"):
                return candidate
        return "custom"
    return parts[0]


def encode_model_choice(provider: str | None, model: str | None) -> str:
    """``provider:model`` so ACP clients keep provider context."""
    raw_model = str(model or "").strip()
    if not raw_model:
        return ""
    raw_provider = str(provider or "").strip().lower()
    return f"{raw_provider}:{raw_model}" if raw_provider else raw_model


@dataclass
class _ModelCatalog:
    """Deduplicated ACP model rows from the inventory + named endpoints.

    Dedupes on the encoded choice id AND a semantic ``provider:model`` id (``ollama`` ==
    ``custom:ollama``). A bare/``custom`` current provider whose base_url matches an ollama
    inventory row is resolved to ``custom:ollama``."""

    normalize_provider: Callable[[str], str]
    current_model: str
    current_choice_provider: str
    current_base_url: str
    # ``custom`` is a billing class only when named endpoints exist to be the identity instead. On a
    # box with none, ``custom`` + ``model.base_url`` IS the routable identity and the row it owns is
    # the honest answer (#72).
    allow_class_identity: bool = False
    models: list[ModelInfo] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    seen_semantic_ids: set[str] = field(default_factory=set)
    empty_authoritative: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.current_choice_provider == "ollama":
            self.current_choice_provider = "custom:ollama"
        self._identity_resolved = (
            self.current_choice_provider not in {"", "custom"}
            or (self.allow_class_identity and bool(self.current_choice_provider))
        )

    @property
    def identity_resolved(self) -> bool:
        """Whether the current provider is a NAME (``custom:<name>``/canonical slug) rather than the
        billing class ``custom``. A row marked ``current`` under an unresolved identity cannot be
        rendered honestly — it would name a category as if it were the endpoint (#72)."""
        return self._identity_resolved

    def semantic(self, provider_id: str) -> str:
        return _semantic_provider(provider_id, self.normalize_provider)

    def add(self, provider_id: str, model_id: str, name: str, description: str) -> None:
        choice_id = encode_model_choice(provider_id, model_id)
        semantic_id = f"{self.semantic(provider_id)}:{model_id}"
        if not choice_id or choice_id in self.seen_ids or semantic_id in self.seen_semantic_ids:
            return
        self.models.append(ModelInfo(model_id=choice_id, name=name, description=description))
        self.seen_ids.add(choice_id)
        self.seen_semantic_ids.add(semantic_id)

    def add_inventory_rows(self, rows: list, provider_label: Callable[[str], str]) -> None:
        for row in rows:
            raw_row_provider = str(row.get("slug") or "").strip().lower()
            row_provider = self.normalize_provider(raw_row_provider)
            row_base_url = str(row.get("api_url") or "").strip().rstrip("/").lower()
            if row.get("native_catalog_empty"):
                self.empty_authoritative.add(raw_row_provider)
            if raw_row_provider in {"ollama", "custom:ollama"} and (
                self.current_base_url and row_base_url == self.current_base_url
            ) and (not self._identity_resolved or (
                self.current_choice_provider == "custom" and self.allow_class_identity
            )):
                # The base_url answers the question: this is the native ollama server, whose rows
                # already carry a name (``custom:ollama``). Accepting the bare class here would both
                # leave those rows unmarked and insert a phantom ``Custom · <model>`` next to them.
                self.allow_class_identity = False
                self.current_choice_provider = "custom:ollama"
                self._identity_resolved = True
            row_models = row.get("models")
            if not row_provider or not isinstance(row_models, (list, tuple)):
                continue
            provider_name = str(row.get("name") or "").strip() or provider_label(row_provider)
            encoded_provider = (
                "custom:ollama" if raw_row_provider == "ollama"
                else raw_row_provider if raw_row_provider.startswith("custom:")
                else row_provider
            )
            for model_entry in row_models:
                if isinstance(model_entry, dict):
                    model_entry = model_entry.get("id") or model_entry.get("model") or model_entry.get("name")
                rendered_model = str(model_entry or "").strip()
                if not rendered_model:
                    continue
                is_current = self.identity_resolved and rendered_model == self.current_model and (
                    self.semantic(encoded_provider) == self.semantic(self.current_choice_provider)
                )
                self.add(
                    encoded_provider, rendered_model, f"{provider_name} · {rendered_model}",
                    f"Provider: {provider_name}" + (" • current" if is_current else ""),
                )

    def add_named_catalogs(self, catalogs: list, current_choice_provider: str) -> None:
        """Named user-defined endpoints (providers: / custom_providers:) are invisible
        to canonical enumeration — append them like the TUI /model picker. An empty
        catalog marks that slug authoritative-empty."""
        for named_slug, named_label, named_catalog in catalogs:
            if not named_catalog:
                self.empty_authoritative.add(str(named_slug).strip().lower())
                continue
            for named_model, named_desc in named_catalog:
                is_current = named_slug.lower() == current_choice_provider and named_model == self.current_model
                parts = [f"Provider: {named_label}", str(named_desc or "").strip(), "current" if is_current else ""]
                self.add(named_slug, named_model, named_model, " • ".join(part for part in parts if part))


def _declares_named_endpoint() -> bool:
    """Whether the config declares a named endpoint at all — independent of listability.

    ``_named_custom_provider_catalogs()`` answers a different question: what could be *enumerated*
    (it drops entries with no ``models:`` whose ``/models`` probe failed or that need an unset
    ``key_env``, and returns ``[]`` on any config/import error). Keying "is ``custom`` merely a
    billing class here" on that made the identity flip with whether a server answered a 1.5 s probe
    (#72 review round 2)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        logger.debug("could not read the config for named endpoints", exc_info=True)
        return False
    if not isinstance(cfg, dict):
        return False
    providers = cfg.get("providers")
    if isinstance(providers, dict) and any(isinstance(v, dict) for v in providers.values()):
        return True
    legacy = cfg.get("custom_providers")
    return isinstance(legacy, list) and any(isinstance(e, dict) for e in legacy)


def build_model_state(model: str, provider: str, base_url: str) -> SessionModelState | None:
    """Picker state from the shared inventory + named endpoints; ``None`` when nothing is listable
    (caller falls back to a single current-model row). Raises on inventory failure."""
    from hermes_cli.inventory import build_models_payload, load_picker_context
    from hermes_cli.models import normalize_provider, provider_label

    normalized_provider = normalize_provider(provider)
    context = load_picker_context().with_overrides(
        current_provider=normalized_provider, current_model=model, current_base_url=base_url,
    )
    payload = build_models_payload(
        context, explicit_only=True, include_unconfigured=False, picker_hints=False,
        canonical_order=True, pricing=False, capabilities=False, refresh=False,
        probe_custom_providers=False, probe_current_custom_provider=False, max_models=ACP_MAX_MODELS_PER_PROVIDER,
    )

    named_catalogs = _named_custom_provider_catalogs()
    named_slugs = {str(slug).strip().lower() for slug, _label, _models in named_catalogs}
    current_choice_provider = str(provider or "").strip().lower()
    current_base = base_url.strip().rstrip("/").lower()
    # ``build_models_payload`` represents configured ``providers:`` entries by their raw
    # config key. ACP ids must instead use the durable ``custom:<key>`` identity so the
    # picker value round-trips through ``parse_model_input``. Only user-defined rows are
    # replaced by the named catalogs: a ``providers:`` key that shadows a canonical name
    # (``providers.openrouter:`` → proxy) must leave the canonical row — and a session that
    # runs on the canonical endpoint — alone, or picking "current" re-routes to the proxy.
    all_rows = payload.get("providers") or []
    canonical_current = any(
        str(r.get("slug") or "").strip().lower() == current_choice_provider and not r.get("is_user_defined")
        for r in all_rows
    )
    inventory_rows: list = []
    for row in all_rows:
        slug = str(row.get("slug") or "").strip().lower()
        if not row.get("is_user_defined") or not {slug, f"custom:{slug}"} & named_slugs:
            inventory_rows.append(row)
            continue
        row_base = str(row.get("api_url") or "").strip().rstrip("/").lower()
        bare = slug.removeprefix("custom:")
        # ``custom`` is the CLASS, not a name: prefixing it yields the non-identity ``custom:custom``,
        # which no catalog row can ever match (#72).
        if bare and bare != "custom" and bare == current_choice_provider and (
            current_base == row_base or not canonical_current
        ):
            current_choice_provider = f"custom:{current_choice_provider}"

    # A caller that could only report the billing class (``custom``) still has to mark the ENTRY the
    # session actually runs on. The inventory already knows every endpoint's ``api_url``, and the
    # named catalogs know every entry's models, so recover the identity here instead of inserting a
    # row named after the category (#72). Unique match only: a base_url or model that several
    # entries share is left unresolved rather than guessed, and an unresolved identity marks nothing.
    if current_choice_provider in {"", "custom"}:
        def _row_base(row: dict) -> str:
            return str(row.get("api_url") or "").strip().rstrip("/").lower()

        def _row_model_ids(row: dict) -> set[str]:
            out: set[str] = set()
            for entry in row.get("models") or []:
                if isinstance(entry, dict):
                    entry = entry.get("id") or entry.get("model") or entry.get("name")
                text = str(entry or "").strip()
                if text:
                    out.add(text)
            return out

        def _durable(row: dict) -> str:
            slug = str(row.get("slug") or "").strip().lower()
            bare = slug.removeprefix("custom:")
            if not bare or bare == "custom":
                return ""  # the class bucket is a billing class, never an identity (#72)
            if slug.startswith("custom:") or row.get("is_user_defined"):
                return f"custom:{bare}"
            return normalize_provider(slug)

        legacy_class_owns_url = False
        if current_base:
            matched = [r for r in all_rows if _row_base(r) == current_base]
            owners = {pid for pid in (_durable(r) for r in matched) if pid}
            # A canonical provider that happens to share the URL (an OpenRouter-compatible relay is
            # the common case) must not be chosen by URL alone: the client would persist
            # ``model.provider: openrouter`` as the global default, i.e. a different credential and
            # none of the entry's extra_headers/api_mode. Only durable ``custom:`` ids may claim it.
            custom_owners = {pid for pid in owners if pid.startswith("custom:")}
            if len(custom_owners) == 1:
                current_choice_provider = custom_owners.pop()
            elif not custom_owners and matched:
                # Only the class bucket lives at this url: the identity IS the legacy custom
                # endpoint, and it answers the question — no model-name guess is allowed after it,
                # or a relay that serves vendor model names (`gpt-4o`) gets reported as that vendor
                # and the client persists it as the global default (#72 review).
                legacy_class_owns_url = True
        if current_choice_provider in {"", "custom"} and model and not legacy_class_owns_url:
            # A canonical provider is never inferred from a model name alone: only durable
            # ``custom:<key>`` identities and named catalogs may claim the model.
            holders = {pid for pid in (_durable(r) for r in all_rows if model in _row_model_ids(r))
                       if pid.startswith("custom:")}
            holders |= {str(slug).strip().lower() for slug, _label, catalog in named_catalogs
                        if any(named_model == model for named_model, _desc in catalog)}
            if len(holders) == 1:
                current_choice_provider = holders.pop()

    # The class bucket (``custom``) is kept alive by the legacy ``model.base_url``. Once the entry is
    # known, listing it again offers a row whose id (``custom:<model>``) resolves back to the class —
    # exactly the route that mis-sends the turn — so it is a duplicate, not a choice (#72).
    if current_choice_provider.startswith("custom:"):
        inventory_rows = [r for r in inventory_rows if str(r.get("slug") or "").strip().lower() != "custom"]

    cat = _ModelCatalog(
        normalize_provider=normalize_provider, current_model=model,
        current_choice_provider=current_choice_provider,
        current_base_url=current_base,
        # No named endpoint is DECLARED ⇒ ``custom`` is the routable identity, not merely a billing
        # class. Keyed on the config, not on what could be enumerated (see _declares_named_endpoint).
        # Either source saying "a named endpoint exists" is enough to demote the class: the config
        # is authoritative (it survives an unlistable endpoint), the enumerated catalogs cover the
        # callers that inject them.
        allow_class_identity=not _declares_named_endpoint() and not named_catalogs,
    )
    cat.add_inventory_rows(inventory_rows, provider_label)
    cat.add_named_catalogs(named_catalogs, current_choice_provider)
    available_models = cat.models

    def empty_applies(provider_id: str) -> bool:
        return _empty_catalog_applies(provider_id, cat.empty_authoritative, normalize_provider)

    if cat.empty_authoritative:
        available_models = [m for m in available_models if not empty_applies(_choice_provider(m.model_id))]

    current_is_empty = empty_applies(cat.current_choice_provider)
    if current_is_empty:
        available_models = [m for m in available_models if " • current" not in str(m.description or "")]
    current_model_id = "" if current_is_empty else encode_model_choice(cat.current_choice_provider, model)
    if current_model_id and not cat.identity_resolved:
        # An id whose provider is the class ``custom`` (or nothing) cannot name a row honestly:
        # inserting it advertises ``Custom · <model>`` as an endpoint of its own, and the client sends
        # that invented id back as an explicit pick, re-persisting the wrong route (#72). Mark
        # nothing — an unmarked list is the honest answer, and it tells the client "no default".
        current_model_id = ""
    elif current_model_id and current_model_id not in {item.model_id for item in available_models}:
        provider_name = provider_label(normalized_provider)
        available_models.insert(0, ModelInfo(
            model_id=current_model_id, name=f"{provider_name} · {model}",
            description=f"Provider: {provider_name} • current",
        ))


    if not available_models and current_is_empty:
        return SessionModelState(available_models=[], current_model_id="")
    if available_models:
        # The first-row fallback is for a session that has no model at all (nothing to mark). A
        # session that HAS one, under an identity we could not resolve, must come back unmarked
        # rather than calling row 0 "current" (#72).
        fallback = "" if model else available_models[0].model_id
        return SessionModelState(
            available_models=available_models,
            current_model_id=current_model_id if current_model_id or current_is_empty else fallback,
        )
    return None
