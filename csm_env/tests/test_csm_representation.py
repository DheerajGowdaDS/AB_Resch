import re

import pytest

from csm_env.builder import CSMEnvironmentRepresentation


ROWS = {
    "customer_case": [{
        "case_id": 1233, "number": "CS001233", "account_id": 10,
        "contact_id": 55, "product_id": 20, "installed_product_id": 91,
        "channel": "email", "priority": "P1", "state": "open",
        "short_description": "Production outage", "assignment_group_id": 7,
        "assigned_to": 17, "escalation": True, "escalation_reason": "impact",
        "reopen_count": 0, "sys_created_on": None, "sys_updated_on": None, "closed_on": None,
    }],
    "account": [{"account_id": 10, "name": "ACME", "account_type": "enterprise", "email_domain": "acme.com", "active": True, "sys_created_on": None, "sys_updated_on": None}],
    "contact": [{"contact_id": 55, "account_id": 10, "portal_user_id": 17, "active": True, "is_primary": True, "sys_created_on": None, "sys_updated_on": None}],
    "product": [{"product_id": 20, "name": "Product-A", "category": "network", "product_price": 1000, "lifecycle_state": "active", "sys_created_on": None, "sys_updated_on": None}],
    "installed_product": [{"installed_product_id": 91, "account_id": 10, "product_id": 20, "location_id": 1, "serial_number": "S-91", "status": "installed", "warranty_end": None, "sys_created_on": None, "sys_updated_on": None}],
    "user": [{"user_id": 17, "first_name": "Jane", "last_name": "Doe", "email": "jane@acme.com", "phone": None, "role": "agent", "location_id": 1, "active": True, "sys_created_on": None, "sys_updated_on": None}],
    "user_group": [{"group_id": 7, "name": "Network Support", "type": "support", "active": True, "sys_created_on": None, "sys_updated_on": None}],
    "location": [{"location_id": 1, "name": "Bengaluru", "plot_no": None, "street": "MG Road", "city": "Bengaluru", "country": "India", "active": True, "sys_created_on": None, "sys_updated_on": None}],
}


class FakeSQL:
    async def fetch_rows(self, query):
        m = re.search(r"FROM\s+(\w+).*?(?:WHERE\s+(\w+)\s*=\s*([^\s;]+))?", query, re.I | re.S)
        table = m.group(1)
        rows = list(ROWS.get(table, []))
        if m.group(2):
            col = m.group(2)
            value = m.group(3).strip("'")
            rows = [r for r in rows if str(r.get(col)) == value]
        return rows


@pytest.mark.asyncio
async def test_case_context_is_connected_and_grounded():
    env = CSMEnvironmentRepresentation(FakeSQL())
    context = await env.get_case_context(1233, max_hops=2)
    assert context["case"]["state"] == "open"
    relations = {r["relation"] for r in context["relations"]}
    assert "BELONGS_TO" in relations
    assert "REPORTED_BY" in relations
    assert "CONCERNS_PRODUCT" in relations
    assert "ASSIGNED_TO" in relations

    ground_truth = await env.get_current_state("customer_case", 1233)
    assert ground_truth["priority"] == "P1"
