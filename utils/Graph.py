class Vertex(object):

    def __init__(self, vertex_id):

        self.id = vertex_id
        self.edges = []
    
class Edge(object):

    def __init__(self, from_vertex, to_vertex, cost=1):

        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.cost = cost

        assert cost >= 0, (f"Cost cannot be negative: {self}, cost={cost}")

    def __repr__(self):

         return f"{self.from_vertex} --> {self.to_vertex}"
         
class Graph(object):

    def __init__(self):

        self.vertices = {}

    def add_vertex(self, vertex):
        
        if(isinstance(vertex, Vertex) and vertex.id not in self.vertices):
            self.vertices[vertex.id] = vertex
            return True
        else:
            return False

    def add_edge(self, edge):

        v1 = edge.from_vertex
        v2 = edge.to_vertex
        
        if(isinstance(edge, Edge)):    
          if(v1 in self.vertices and v2 in self.vertices):
              self.vertices[v1].edges.append(edge)

              rEdge = Edge(v2, v1)
              self.vertices[v2].edges.append(rEdge)
              return True
          else:
              return False

    def get_vertices(self):

        return self.vertices.keys()

    def __iter__(self):

        return iter(self.vertices.values())

    def dfs(self, start_vertex_id, visited=set()):

        start_vertex = self.vertices.get(start_vertex_id)

        if not start_vertex:
           return []

        dfs_order = []
        
        visited.add(start_vertex_id)
        dfs_order.append(start_vertex_id)

        for neighbor in start_vertex.edges:
            if neighbor.to_vertex not in visited:
               dfs_order.extend(self.dfs(neighbor.to_vertex, visited.copy()))

        return dfs_order
