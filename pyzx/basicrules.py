# PyZX - Python library for quantum circuit rewriting 
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains several rules more easily applied interactively to ZX
diagrams. The emphasis is more on ease of use and simplicity than performance.

Rules are given as functions that take as input a vertex or a pair of vertices
to fix the location the rule is applied. They then apply the rule and return
True if the rule indeed applies at this location, otherwise they return false.

Most rules also have a companion function check_RULENAME, which only checks
whether the rule applies at the given location and doesn't actually apply
the rule.
"""

__all__ = ['color_change_diagram',
        'check_color_change',
        'color_change',
        'check_copy_X',
        'copy_X',
        'check_copy_Z',
        'copy_Z',
        'check_pi_commute_X',
        'pi_commute_X',
        'check_pi_commute_Z',
        'pi_commute_Z',
        'check_strong_comp',
        'strong_comp',
        'check_fuse',
        'fuse',
        'check_remove_id',
        'remove_id']

from typing import Tuple, List
from .graph.base import BaseGraph, VT, ET
from .utils import VertexType, EdgeType

def color_change_diagram(g: BaseGraph[VT,ET]):
    """Color-change an entire diagram by applying Hadamards to the inputs and ouputs."""
    for v in g.vertices():
        if g.type(v) == VertexType.BOUNDARY:
            if g.vertex_degree(v) != 1:
                raise ValueError("Boundary should only have 1 neighbor.")
            v1 = next(iter(g.neighbors(v)))
            e = g.edge(v,v1)
            g.set_edge_type(e, EdgeType.SIMPLE
                    if g.edge_type(e) == EdgeType.HADAMARD
                    else EdgeType.HADAMARD)
        elif g.type(v) == VertexType.Z:
            g.set_type(v, VertexType.X)
        elif g.type(v) == VertexType.X:
            g.set_type(v, VertexType.Z)

def check_color_change(g: BaseGraph[VT,ET], v: VT) -> bool:
    if not (g.type(v) == VertexType.Z or g.type(v) == VertexType.X):
        return False
    else:
        return True

def color_change(g: BaseGraph[VT,ET], v: VT) -> bool:
    if not (g.type(v) == VertexType.Z or g.type(v) == VertexType.X):
        return False

    g.set_type(v, VertexType.Z if g.type(v) == VertexType.X else VertexType.X)
    for v1 in g.neighbors(v):
        e = g.edge(v,v1)
        g.set_edge_type(e, EdgeType.SIMPLE
                if g.edge_type(e) == EdgeType.HADAMARD
                else EdgeType.HADAMARD)

    return True

def check_strong_comp(g: BaseGraph[VT,ET], v1: VT, v2: VT) -> bool:
    if not (((g.type(v1) == VertexType.X and g.type(v2) == VertexType.Z) or
             (g.type(v1) == VertexType.Z and g.type(v2) == VertexType.X)) and
            (g.phase(v1) == 0 or g.phase(v1) == 1) and
            (g.phase(v2) == 0 or g.phase(v2) == 1) and
            g.connected(v1,v2) and
            g.edge_type(g.edge(v1,v2)) == EdgeType.SIMPLE):
        return False
    return True

def strong_comp(g: BaseGraph[VT,ET], v1: VT, v2: VT) -> bool:
    if not check_strong_comp(g, v1, v2): return False    
    
    nhd: Tuple[List[VT],List[VT]] = ([],[])
    v = (v1,v2)

    for i in range(2):
        j = (i + 1) % 2
        for vn in g.neighbors(v[i]):
            if vn != v[j]:
                q = 0.4*g.qubit(vn) + 0.6*g.qubit(v[i])
                r = 0.4*g.row(vn) + 0.6*g.row(v[i])
                newv = g.add_vertex(g.type(v[j]), qubit=q, row=r)
                g.add_edge(g.edge(newv,vn), edgetype=g.edge_type(g.edge(v[i],vn)))
                g.set_phase(newv, g.phase(v[j]))
                nhd[i].append(newv)

    for n1 in nhd[0]:
        for n2 in nhd[1]:
            g.add_edge(g.edge(n1,n2))

    g.scalar.add_power((len(nhd[0]) - 1) * (len(nhd[1]) - 1))
    if g.phase(v1) == 1 and g.phase(v2) == 1:
        g.scalar.add_phase(1)

    g.remove_vertex(v1)
    g.remove_vertex(v2)
    
    return True

def check_copy_X(g: BaseGraph[VT,ET], v: VT) -> bool:
    if not (g.vertex_degree(v) == 1 and
            g.type(v) == VertexType.X and
            (g.phase(v) == 0 or g.phase(v) == 1)):
        return False
    nv = next(iter(g.neighbors(v)))
    if not (g.type(nv) == VertexType.Z and
            g.edge_type(g.edge(v,nv)) == EdgeType.SIMPLE):
        return False
    return True

def copy_X(g: BaseGraph[VT,ET], v: VT) -> bool:
    if not check_copy_X(g, v): return False    
    nv = next(iter(g.neighbors(v)))
    strong_comp(g, v, nv)
    
    return True

def check_pi_commute_Z(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) == VertexType.Z

def pi_commute_Z(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_pi_commute_Z(g, v): return False
    g.set_phase(v, -g.phase(v))
    ns = g.neighbors(v)
    for w in ns:
        e = g.edge(v, w)
        et = g.edge_type(e)
        if ((g.type(w) == VertexType.Z and et == EdgeType.HADAMARD) or
            (g.type(w) == VertexType.X and et == EdgeType.SIMPLE)):
            g.add_to_phase(w, 1)
        else:
            g.remove_edge(e)
            c = g.add_vertex(VertexType.X,
                    qubit=0.5*(g.qubit(v) + g.qubit(w)),
                    row=0.5*(g.row(v) + g.row(w)))
            g.add_edge(g.edge(v, c))
            g.add_edge(g.edge(c, w), edgetype=et)
    return True
    
def check_pi_commute_X(g: BaseGraph[VT,ET], v: VT) -> bool:
    color_change_diagram(g)
    b = check_pi_commute_Z(g, v)
    color_change_diagram(g)
    return b

def pi_commute_X(g: BaseGraph[VT,ET], v: VT) -> bool:
    color_change_diagram(g)
    b = pi_commute_Z(g, v)
    color_change_diagram(g)
    return b

def check_copy_Z(g: BaseGraph[VT,ET], v: VT) -> bool:
    color_change_diagram(g)
    b = check_copy_X(g, v)
    color_change_diagram(g)
    return b

def copy_Z(g: BaseGraph, v: VT) -> bool:
    color_change_diagram(g)
    b = copy_X(g, v)
    color_change_diagram(g)
    return b

def check_fuse(g: BaseGraph[VT,ET], v1: VT, v2: VT) -> bool:
    if not (g.connected(v1,v2) and
            ((g.type(v1) == VertexType.Z and g.type(v2) == VertexType.Z) or
             (g.type(v1) == VertexType.X and g.type(v2) == VertexType.X)) and
            g.edge_type(g.edge(v1,v2)) == EdgeType.SIMPLE):
        return False
    else:
        return True

def fuse(g: BaseGraph[VT,ET], v1: VT, v2: VT) -> bool:
    if not check_fuse(g, v1, v2): return False
    g.add_to_phase(v1, g.phase(v2), g.get_params(v2))
    for v3 in g.neighbors(v2):
        if v3 != v1:
            g.add_edge_smart(g.edge(v1,v3), edgetype=g.edge_type(g.edge(v2,v3)))
    
    g.remove_vertex(v2)
    return True

def check_remove_id(g: BaseGraph[VT,ET], v: VT) -> bool:
    if not (g.vertex_degree(v) == 2 and g.phase(v) == 0):
        return False
    else:
        return True

def remove_id(g: BaseGraph[VT,ET], v: VT) -> bool:
    if not check_remove_id(g, v):
        return False
    
    v1, v2 = tuple(g.neighbors(v))
    g.add_edge_smart(g.edge(v1,v2), edgetype=EdgeType.SIMPLE
            if g.edge_type(g.edge(v,v1)) == g.edge_type(g.edge(v,v2))
            else EdgeType.HADAMARD)
    g.remove_vertex(v)
    
    return True


