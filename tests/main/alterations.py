from src.main import model, nodes, alterations

m = model.Model(4, 4)
ln = nodes.Linear('relu', 0.3)
m.add_node(ln)
alt = alterations.AlterationV2(m)
alt.add_connection(0, 8)
alt.add_connection(1, 8)
alt.add_connection(8, 7)
alt.add_connection(8, 6)
alt.make_graph(None, alt.io.nodes)