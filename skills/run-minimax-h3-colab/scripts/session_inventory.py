from __future__ import annotations

import json

from colab_cli.common import state


assignments = state.client.list_assignments()
print(json.dumps({"endpoints": [assignment.endpoint for assignment in assignments]}))
