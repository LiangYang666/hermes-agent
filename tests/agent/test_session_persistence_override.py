"""The persist override is the clean user row even when runtime content is a plain-text list.

ACP surface notes ride the api_content sidecar and leave a string override behind; a list of
text parts is not native media, so the override must win and the durable row must stay clean.
"""
from types import SimpleNamespace

from agent.session_persistence import durable_user_row_content


def test_plain_text_list_content_takes_the_string_override():
    content = [{'type': 'text', 'text': 'hello'}]
    ingested, api_content = durable_user_row_content(
        SimpleNamespace(_persist_user_message_override='hello'), {}, content, None)
    assert ingested == 'hello'
    assert api_content is None  # no sidecar needed: the list carried no native parts


def test_native_media_list_content_still_refuses_a_plain_string_override():
    content = [{'type': 'text', 'text': 'look'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png,x'}}]
    ingested, api_content = durable_user_row_content(
        SimpleNamespace(_persist_user_message_override='look'), {}, content, None)
    assert ingested is content
    assert api_content is None
