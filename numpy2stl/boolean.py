import numpy as np
import pymeshlab as ml

def union_pymesh(models):

	ms = ml.MeshSet()
	
	for key in models:
		vx,fs = models[key]
		vx = vx.astype(np.float32)
		fs = fs.astype(np.int32)
		mesh = ml.Mesh(vertex_matrix=vx, face_matrix=fs)
		ms.add_mesh(mesh, key)

	ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)

	result = ms.current_mesh()
	vx, fs = result.vertex_matrix(), result.face_matrix()

	return vx, fs


def cut_puzzle_pieces(model, puzzle):

	pieces_out = {}
	
	for key in puzzle:
		print(key)
		ms = ml.MeshSet()
		###
		vx,fs = model
		vx = vx.astype(np.float32).copy()
		fs = fs.astype(np.int32).copy()
		mesh = ml.Mesh(vertex_matrix=vx, face_matrix=fs)
		ms.add_mesh(mesh, "model")

		###
		vx,fs = puzzle[key]
		vx = vx.astype(np.float32).copy()
		fs = fs.astype(np.int32).copy()
		mesh = ml.Mesh(vertex_matrix=vx, face_matrix=fs)
		ms.add_mesh(mesh, str(key))

		###
		try:
			ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)
		except Exception as E:
			print(E)

		result = ms.current_mesh()
		vx, fs = result.vertex_matrix(), result.face_matrix()

		if len(vx)==0:
			continue

		pieces_out[key] = vx, fs

	return pieces_out