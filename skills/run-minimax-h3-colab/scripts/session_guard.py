from __future__ import annotations

import json
import sys

from colab_cli.commands.session import spawn_keep_alive
from colab_cli.common import state
from colab_cli.state import SessionState


name = sys.argv[1]
sessions, assignments = state.sync_sessions()
if name in sessions:
    session = sessions[name]
    print(json.dumps({"action": "reuse", "endpoint": session.endpoint}))
elif not assignments:
    print(json.dumps({"action": "create"}))
elif len(assignments) == 1:
    assignment = assignments[0]
    info = assignment.runtime_proxy_info
    variant_value = getattr(assignment, "variant", None)
    accelerator_value = getattr(assignment, "accelerator", None)
    variant = str(getattr(variant_value, "name", variant_value or "GPU"))
    accelerator = str(getattr(accelerator_value, "value", accelerator_value or "G4"))
    session = SessionState(
        name=name,
        token=info.token,
        url=info.url,
        endpoint=assignment.endpoint,
        variant=variant,
        accelerator=accelerator,
    )
    state.store.add(session)
    session.keep_alive_pid = spawn_keep_alive(
        assignment.endpoint,
        name,
        auth_provider=state.auth_provider,
        config_path=state.config_path,
    )
    state.store.add(session)
    print(json.dumps({"action": "rebind", "endpoint": assignment.endpoint}))
else:
    print(json.dumps({
        "action": "ambiguous",
        "endpoints": [assignment.endpoint for assignment in assignments],
    }))
