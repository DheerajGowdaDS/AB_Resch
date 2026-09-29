from csm_env.graph import Edge, EnvironmentGraph, Node


def test_graph_neighbors_and_subgraph():
    g = EnvironmentGraph()
    g.add_node(Node("1", "CustomerCase", "customer_case", {"state": "open"}))
    g.add_node(Node("10", "Account", "account", {"name": "ACME"}))
    g.add_edge(Edge("CustomerCase:1", "Account:10", "BELONGS_TO"))

    neighbors = g.neighbors("CustomerCase", "1")
    assert len(neighbors) == 1
    assert neighbors[0][0].type == "Account"
    assert neighbors[0][1].relation == "BELONGS_TO"

    sub = g.subgraph_for("CustomerCase", "1", max_hops=1)
    assert len(sub.nodes) == 2
