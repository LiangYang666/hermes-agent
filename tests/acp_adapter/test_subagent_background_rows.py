"""子代理到达编辑器的形状（issue #97）。

上游把 ``subagent.*`` 只发给 gateway 自己的事件流；ACP 侧以前一条都不投影，于是驾驶舱只能看见 shell
进程，看不见被派出去的子代理（而它从前是有这一类的）。这里钉住三件事：

* 生命周期翻成 ``kind: "delegation"`` 的行，与 ``background.tasks`` 同形状；
* 结束状态带 ``exitCode``（失败就非 0）——「成功还是失败」是判定，不是形容词；
* 后到的进程快照不会把子代理那几行整片盖掉（两套词汇合成一帧）。
"""
import pytest
from unittest.mock import AsyncMock

from acp_adapter.gateway_server import GatewayACPAgent


def _agent():
    agent = GatewayACPAgent()
    agent._gateway, agent._conn = AsyncMock(), AsyncMock()
    agent._snapshots['s'] = {'execution_generation': 1}
    return agent


def _event(kind, **payload):
    return {'session_id': 's', 'type': kind, 'payload': payload,
            'execution_generation': 1, 'seq': 1}


def _rows(agent):
    return agent._conn.session_update.await_args.kwargs['update'].field_meta['hermes']['tasks']


@pytest.mark.asyncio
async def test_a_running_subagent_is_a_delegation_row():
    agent = _agent()
    await agent._project(_event('subagent.spawn_requested', subagent_id='sa1',
                                goal='research the ACP surface', task_index=0, task_count=2))
    rows = _rows(agent)
    assert len(rows) == 1
    row = rows[0]
    assert row['kind'] == 'delegation'
    assert row['status'] == 'running'
    assert row['command'] == 'research the ACP surface'
    assert row['taskCount'] == 2
    # 一个子代理没有 pid：它不是进程，硬塞一个会让界面画出「pid ?」。
    assert row['pid'] is None
    assert row['notifyOnComplete'] is True


@pytest.mark.asyncio
async def test_completion_carries_a_real_exit_code_and_keeps_the_row():
    agent = _agent()
    await agent._project(_event('subagent.start', subagent_id='sa1', goal='g'))
    await agent._project(_event('subagent.complete', subagent_id='sa1', goal='g',
                                status='failed', duration_seconds=12.4, summary='boom'))
    row = _rows(agent)[0]
    assert row['status'] == 'exited'
    assert row['exitCode'] == 1
    assert row['uptimeSeconds'] == 12
    assert row['phase'] == 'failed'

    await agent._project(_event('subagent.complete', subagent_id='sa2', goal='h',
                                status='completed', duration_seconds=3))
    rows = _rows(agent)
    assert [r['exitCode'] for r in rows] == [1, 0]
    assert [r['command'] for r in rows] == ['g', 'h']


@pytest.mark.asyncio
async def test_a_process_snapshot_does_not_wipe_the_subagents():
    agent = _agent()
    await agent._project(_event('subagent.start', subagent_id='sa1', goal='g'))
    processes = [{'id': 'p1', 'command': 'npm run build', 'pid': 42, 'status': 'running',
                  'uptimeSeconds': 3, 'notifyOnComplete': True, 'kind': 'process'}]
    await agent._project(_event('background.tasks', tasks=processes))
    rows = _rows(agent)
    assert [r['id'] for r in rows] == ['p1', 'subagent:sa1']


@pytest.mark.asyncio
async def test_narration_events_do_not_republish_a_frame():
    """``subagent.progress`` 每标出一行；每条都重发整表就是刷屏，而它不改变任何一行。"""
    agent = _agent()
    await agent._project(_event('subagent.progress', subagent_id='sa1', goal='g'))
    assert agent._conn.session_update.await_count == 0
