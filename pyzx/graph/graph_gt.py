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

import graph_tool.all as gt

from .base import BaseGraph

#DEPRECATED
class GraphGT(BaseGraph):
	backend = 'graph_tool'
	def __init__(self):
		raise Warning("Graph_tool is currently not fully supported.")
		BaseGraph.__init__(self)
		self.graph = gt.Graph(directed=False)
		self.graph.set_fast_edge_removal()
		self.graph.vp.type = self.graph.new_vertex_property('int')

	def add_vertices(self, amount):
		self.graph.add_vertex(amount)

	def add_edges(self, edges):
		self.graph.add_edge_list(edges)

	def remove_vertices(self, vertices):
		for v in reversed(sorted(vertices)):
			self.graph.remove_vertex(v,fast=True)

	def remove_vertex(self, vertex):
		self.graph.remove_vertex(v)

	def remove_edges(self, edges):
		for e in edges:
			if isinstance(e,tuple):
				self.graph.remove_edge(self.graph.edge(e[0],e[1]))
			else:
				self.graph.remove_edge(e)

	def num_vertices(self):
		return self.graph.get_vertices().shape[0]

	def num_edges(self):
		return self.graph.get_edges().shape[0]

	def vertices(self):
		'''Iterator over all the vertices'''
		return self.graph.vertices()

	def verts_as_int(self, verts):
		return [int(v) for v in verts]

	def edges(self):
		'''Iterator over all the edges'''
		return self.graph.edges()

	def edges_as_int(self, edges):
		return [(self.graph.edge_index[e] if isinstance(e,gt.Edge) else e) for e in edges]

	def edge_set(self):
		#return set(self.graph.get_edges()[...,2])
		return set(self.graph.edges())

	def edge_st(self, edge):
		'''Returns a tuple of source/target of the given edge'''
		if isinstance(edge, gt.Edge):
			return (edge.source(), edge.target())
		e = gt.find_edge(self.graph,self.graph.edge_index,edge)[0]
		return (e.source(), e.target())

	def get_neighbors(self, vertex):
		'''Returns a tuple of source/target of the given edge'''
		return vertex.all_neighbors()

	def get_vertex_degree(self, vertex):
		'''Returns all neighboring vertices of the given vertex'''
		return vertex.in_degree() + vertex.out_degree()

	def get_incident_edges(self, vertex):
		#return [self.graph.edge_index[e] for e in vertex.all_edges()]
		return vertex.all_edges()

	def is_connected(self,v1,v2):
		return bool(self.graph.edge(v1, v2))

	def is_equal(self,v1,v2):
		'''Returns whether v1 and v2 represent the same vertex'''
		return int(v1) == int(v2)

	def get_type(self, vertex):
		return self.graph.vp.type[vertex]

	def set_type(self, vertex, t):
		self.graph.vp.type[vertex] = t

	def add_attribute(self, attrib_name, default=0):
		self.graph.vertex_properties[attrib_name] = self.graph.new_vertex_property('int')
		self.graph.vertex_properties[attrib_name].set_value(default)
