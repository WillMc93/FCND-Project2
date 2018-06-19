import numpy as np

def point(p):
	return np.array([p[0], p[1], 1]).reshape(1, -1)

def collinear(p1, p2, p3, epsilon=1e-6):
	m = np.concatenate((p1, p2, p3), 0)
	det = np.linalg.det(m)
	return abs(det) < epsilon

def collinearity_prune(path):
	""" This is the basic collinear pruning algorithm from class. I quit using
		in favour of var_prune below (which is mostly the same thing). I'm leaving this
		here just to know where we've been. """
	pruned = [p for p in path]
	i = 0
	while i < len(pruned) - 2:
		p1 = point(pruned[i])
		p2 = point(pruned[i+1])
		p3 = point(pruned[i+2])

		# If all three points are collinear, we can remove the middle point and
		# have a working path between points p1 and p3
		if collinear(p1, p2, p3):
			pruned.remove(pruned[i+1])
		else:
			i += 1

	return pruned

def var_prune(path, point_range=5):
	""" This is collinear pruning algorithm that considers a variable range of points.
	 	This is useful for eliminating zig-zags in the path ("simple anti-aliasing").
		This algorithm can do the same thing as collinearity_prune with point_range=3. """
	pruned = [p for p in path]
	i = 0

	# Check parameters and adjust as necessary
	if point_range < 3:
		# Can't be less than 3 or the collinearity doesn't work
		point_range = 3
	if point_range % 2 == 0:
		# Always want to consider an odd number of points
		point_range += 1

	# Since we are wanting to have a variable range to consider for collinearity,
	# we need to define how big of a step we need between points in the point_range.
	step = 1 # step == 1 is fine if we are only considering 3 points
	if point_range > 3:
		step = int(np.ceil(0.5 * (point_range - 1)))

	# Gather range of offsets to consider (ie 0, 2, 4 for the default case {num_steps = 5})
	offsets = range(0, point_range, step)
	while i < len(pruned) - point_range - 1:
		# Select points to consider
		p1 = point(pruned[i + offsets[0]]) # NOTE: offsets[0] should always be 0
		p2 = point(pruned[i + offsets[1]])
		p3 = point(pruned[i + offsets[2]])

		# If all three points are collinear, we can remove the middle points and
		# have a working path between points p1 and p3
		if collinear(p1, p2, p3):
			to_remove = range(1, offsets[2])
			for tr in to_remove:
				pruned.remove(pruned[i + tr])
		else:
			i += 1

	return pruned

def run_prune(path):
	""" After getting rid of the zig-zags, it looked like more points were
		collinear, so here I run the two a couple of times to straighten
		everything out. """
	pruned = False
	new_path = []
	print("Pruning ...")
	while not pruned:
		# Run pruning algo with variable ranges
		col_path = var_prune(path, point_range=3) # col_path == "collinear path"
		aa_path = var_prune(col_path, point_range=5) # aa_path == "anti-aliased"

		# If a path exists and is shorter, that is the one we want.
		if len(aa_path) > 0 and len(aa_path) < len(col_path):
			new_path = aa_path
		elif len(col_path) > 0 and len(col_path) < len(path):
			new_path = col_path
		# Keep output clean
		if len(path) != len(new_path):
			print("Pruned from {} waypoints to {} waypoints.".format(len(path), len(new_path)))

		# Run until nothing is removed
		if len(new_path) == len(path):
			pruned = True

		path = new_path

	return path
