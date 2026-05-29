# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import patch
from aidweather.client import PowerClient


@pytest.fixture(autouse=True)
def cleanup_power_clients():
    """Automatically tracks and closes all PowerClient database connections created during a test to prevent ResourceWarnings."""
    clients = []

    # Store the original __init__
    original_init = PowerClient.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        clients.append(self)

    with patch.object(PowerClient, "__init__", patched_init):
        yield

    for client in clients:
        if hasattr(client, "db_conn") and client.db_conn:
            try:
                client.db_conn.close()
            except Exception:
                pass
            client.db_conn = None
