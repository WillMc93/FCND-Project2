import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid
from pruning_utils import prune
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

""" Student Imports """
import csv # because lazy/dumb (probably don't need the overhead to do this)

class States(Enum):
	MANUAL = auto()
	ARMING = auto()
	TAKEOFF = auto()
	WAYPOINT = auto()
	LANDING = auto()
	DISARMING = auto()
	PLANNING = auto()


class MotionPlanning(Drone):

	def __init__(self, connection):
		super().__init__(connection)

		self.target_position = np.array([0.0, 0.0, 0.0])
		self.waypoints = []
		self.in_mission = True
		self.check_state = {}

		# initial state
		self.flight_state = States.MANUAL

		# register all your callbacks here
		self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
		self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
		self.register_callback(MsgID.STATE, self.state_callback)

	def local_position_callback(self):
		if self.flight_state == States.TAKEOFF:
			if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
				self.waypoint_transition()
		elif self.flight_state == States.WAYPOINT:
			if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
				if len(self.waypoints) > 0:
					self.waypoint_transition()
				else:
					if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
						self.landing_transition()

	def velocity_callback(self):
		if self.flight_state == States.LANDING:
			if self.global_position[2] - self.global_home[2] < 0.1:
				if abs(self.local_position[2]) < 0.01:
					self.disarming_transition()

	def state_callback(self):
		if self.in_mission:
			if self.flight_state == States.MANUAL:
				self.arming_transition()
			elif self.flight_state == States.ARMING:
				if self.armed:
					self.plan_path()
			elif self.flight_state == States.PLANNING:
				self.takeoff_transition()
			elif self.flight_state == States.DISARMING:
				if ~self.armed & ~self.guided:
					self.manual_transition()

	def arming_transition(self):
		self.flight_state = States.ARMING
		print("arming transition")
		self.arm()
		self.take_control()

	def takeoff_transition(self):
		#pdb.set_trace()
		self.flight_state = States.TAKEOFF
		print("takeoff transition")
		self.takeoff(10)

	def waypoint_transition(self):
		self.flight_state = States.WAYPOINT
		print("waypoint transition")
		self.target_position = self.waypoints.pop(0)
		self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

	def landing_transition(self):
		self.flight_state = States.LANDING
		print("landing transition")
		self.land()

	def disarming_transition(self):
		self.flight_state = States.DISARMING
		print("disarm transition")
		self.disarm()
		self.release_control()

	def manual_transition(self):
		self.flight_state = States.MANUAL
		print("manual transition")
		self.stop()
		self.in_mission = False

	def send_waypoints(self):
		print("Sending waypoints to simulator ...")
		data = msgpack.dumps(self.waypoints)
		self.connection._master.write(data)

	def plan_path(self, goal_lon=-122.39735, goal_lat=37.79737):
		self.flight_state = States.PLANNING
		print("Searching for a path ...")
		TARGET_ALTITUDE = 10
		SAFETY_DISTANCE = 5

		self.target_position[2] = TARGET_ALTITUDE

		# read lat0, lon0 from colliders into floating point values
		lat0, lon0 = None, None
		with open('colliders.csv') as colliders:
			reader = csv.reader(colliders)
			(lat0, lon0) = next(reader)
			# Trim leading label and convert sting to float
			lat0 = float(lat0[5:])
			lon0 = float(lon0[5:])

		# set home position to (lon0, lat0, 0)
		self.set_home_position(lon0, lat0, 0)

		# convert self.global_position to current local position using global_to_local()
		local_pos = global_to_local(self.global_position, self.global_home)

		print('global home {0} \nposition {1} \nlocal position {2}'.format(self.global_home, self.global_position,
																		 self.local_position))
		# Read in obstacle map
		data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

		# Define a grid for a particular altitude and safety margin around obstacles
		grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
		print("North offset = {0}, East offset = {1}".format(north_offset, east_offset))

		# Define starting point on the grid
		grid_north = int(np.ceil(local_pos[0] - north_offset))
		grid_east = int(np.ceil(local_pos[1] - east_offset))
		grid_start = (grid_north, grid_east)

		# Set goal as some arbitrary position on the grid
		grid_goal = (None, None)
		if goal_lon == None or goal_lat == None:
			grid_goal = (-north_offset + 10, -east_offset + 10)
		# Set goal as latitude / longitude position and convert
		else:
			# Get local coordinates from goal lon and lat
			""" Need to convert to list here or grid_goal values will end up
				getting converted to a float when casting to a tuple."""
			grid_goal = list(global_to_local((goal_lon, goal_lat, TARGET_ALTITUDE), self.global_home)[0:2])
			# Apply Offsets and convert to tuple
			grid_goal[0] = int(np.ceil(grid_goal[0] - north_offset))
			grid_goal[1] = int(np.ceil(grid_goal[1] - east_offset))
			grid_goal = tuple(grid_goal)

		# Run A* to find a path from start to goal
		print('Local Start and Goal: ', grid_start, grid_goal)
		path, _ = a_star(grid, heuristic, grid_start, grid_goal)
		# prune path to minimize number of waypoints
		path = run_prune(path)

		# Convert path to waypoints
		waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
		# Set self.waypoints
		self.waypoints = waypoints

		self.send_waypoints()

	def start(self):
		self.start_log("Logs", "NavLog.txt")

		print("starting connection")
		self.connection.start()

		# Only required if they do threaded
		# while self.in_mission:
		#    pass

		self.stop_log()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int, default=5760, help='Port number')
	parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
	args = parser.parse_args()

	conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
	drone = MotionPlanning(conn)
	time.sleep(1)

	drone.start()
