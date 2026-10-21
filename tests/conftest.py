"""shared test setup: the world the tests run against"""

import pytest

from logistics_agent.domain import Shipment
from logistics_agent.reasoner import Reasoner

# the starting world has no shipments; the tests add one, the crate, as a human would before asking for an estimate
CRATE = Shipment(name="shipment1", details="crate of machine parts", weight=1000)

# the world graph as the reasoner builds it, without the crate
configure_world_graph_without_shipments = Reasoner.configure_world_graph


@pytest.fixture(autouse=True)
def world_with_the_crate(monkeypatch):
    """every world graph the tests build has the crate in it, nowhere until the planner's current facts place it"""

    def configure_world_graph():
        world = configure_world_graph_without_shipments()
        world.add_entity(CRATE)
        return world

    monkeypatch.setattr(Reasoner, "configure_world_graph", staticmethod(configure_world_graph))


@pytest.fixture
def starting_world():
    """the world graph as the reasoner builds it, without the crate the tests add"""
    return configure_world_graph_without_shipments()
