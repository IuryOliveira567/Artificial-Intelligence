class Vertex(object):

    def __init__(self, vertex_id):

        self.id = vertex_id
        self.neighbors = set()

    def add_neighbor(self, neighbor):

        if(neighbor not in self.neighbors):
            self.neighbors.add(neighbor)

class Graph(object):

    def __init__(self):

        self.vertices = {}

    def add_vertex(self, vertex):

        if(isinstance(vertex, Vertex) and vertex.id not in self.vertices):
            self.vertices[vertex.id] = vertex
            return True
        else:
            return False

    def add_edge(self, v1, v2):

        if(v1 in self.vertices and v2 in self.vertices):
            self.vertices[v1].add_neighbor(v2)
            self.vertices[v2].add_neighbor(v1)
            return True
        else:
            return False

    def get_vertices(self):

        return self.vertices.keys()

    def __iter__(self):

        return iter(self.vertices.values())

   
    
