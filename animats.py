#!/usr/bin/env python

"""
14F - COM SCI 263C
Eric Marcin-Cuddy (203287164)
"""

################################################################################
### modules and libraries

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from pybrain.rl.agents import LearningAgent
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task
from pybrain.rl.experiments import Experiment
from pybrain.rl.explorers import EpsilonGreedyExplorer
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable

import numpy as np
from scipy.spatial.distance import euclidean

import random

################################################################################
### constants

DIMS            = np.array([10,10,10])
COLOR_MAX       = 1
COLOR_THRESHOLD = 0.125 * COLOR_MAX
DIST_THRESHOLD  = 1.0
DIST_MAX        = max(DIMS)
BKGD_COLOR      = np.random.rand(3) * COLOR_THRESHOLD
MAX_TIME        = 1000
LIFE_UNIT       = 0.1
ALPHA           = 0.5
GAMMA           = 0.0
EPSILON         = 0.0

################################################################################
### classes for Q-learning

class PredatorTask(Task):

  def __init__(self, environment):
    self.env = environment

  def getObservation(self):
    return self.env.getSensors()

  def performAction(self, action):
    self.env.performAction(action)

  def getReward(self):
    return self.env.interaction.getReward()

class PredatorEnvironment(Environment):

  def __init__(self, world):
    self.interaction = PredatorInteraction(world)

  def getSensors(self):
    return self.interaction.getSensors()

  def performAction(self, action):
    self.interaction.performAction(action)

################################################################################
### animat classes

class Animat(object):

  def __init__(self, color, poison):
    self.color    = color
    self.poison   = poison
    self.coords   = np.random.rand(len(DIMS)) * DIMS
    self.life     = 1.0
    self.history  = {}

  def respawn(self, t):
    self.make_history(t, death=True)
    self.coords = np.random.rand(len(DIMS)) * DIMS
    self.life   = 1.0

  def step(self, t):
    move = np.random.rand(len(DIMS)) * np.random.choice([-1,0,1], len(DIMS))
    self.coords += move

  def make_history(self, t, **kwargs):
    if not self.history.has_key(t):
      self.history[t] = {}
    self.history[t].update(kwargs)

class PredatorAnimat(Animat):

  def __init__(self):
    Animat.__init__(self, [COLOR_MAX, 0, 0], 0.0)
    self.eaten    = None
    self.rejected = None

  def eat(self, t, prey):
    self.eaten = prey
    self.make_history(t, ate=type(prey).__name__)
    self.eaten.make_history(t, eaten=type(self).__name__)

  def reject(self, t, prey):
    self.rejected = prey
    self.make_history(t, rejected=type(prey).__name__)
    self.rejected.make_history(t, rejected=type(self).__name__)

  def respawn(self, t):
    Animat.respawn(self, t)
    self.eaten    = None
    self.rejected = None

class PreyAnimat(Animat):

  def __init__(self):
    Animat.__init__(self, [0, COLOR_MAX, 0], 0.0)

class AposematismAnimat(Animat):

  def __init__(self):
    poison = max(0.5, np.random.rand())
    Animat.__init__(self, [0, 0, COLOR_MAX * poison], poison)

class CrypsisAnimat(Animat):

  def __init__(self):
    Animat.__init__(self, np.random.rand(3) * COLOR_THRESHOLD * 2, 0.0)

class MimcryAnimat(Animat):

  def __init__(self, init_color):
    Animat.__init__(self, init_color, 0.0)

################################################################################
### interactions (how I get information from place to place)

class WorldInteraction(object):

  def __init__(self):

    self.predator = PredatorAnimat()
    self.animats  = []
    self.t        = 0

    for x in xrange(3):
      self.animats.append(PreyAnimat())
      self.animats.append(AposematismAnimat())
      self.animats.append(CrypsisAnimat())
      self.animats.append(PredatorAnimat())

class PredatorInteraction(object):

  NSTATES = (COLOR_MAX + 1) ** 3
  ACTIONS = ('EAT', 'REJECT')

  def __init__(self, world):

    self.world = world

  def getSensors(self):
    """
    Observe the color of the nearest animat.
    """

    nearest_prey = min(
      self.world.animats,
      key=lambda animat: \
        euclidean(self.world.predator.coords, animat.coords)
    )

    print 'Nearest prey: %s (%s)' % (nearest_prey, nearest_prey.__dict__)

    if euclidean(self.world.predator.coords, nearest_prey.coords) > DIST_MAX:
      print 'Predator is too far away from the nearest prey!'
      relative_color = BKGD_COLOR

    else:
      relative_color = abs(nearest_prey.color - BKGD_COLOR)

    print 'Relative color: %s' % relative_color

    sensors = [
      relative_color[2] + \
      relative_color[1] * (COLOR_MAX + 1) + \
      relative_color[0] * (COLOR_MAX + 1) ** 2
    ]

    print 'Sensors color: %s' % sensors

    return sensors

  def performAction(self, action):
    """
    Perform the chosen action.
    """

    action = PredatorInteraction.ACTIONS[int(action[0])]

    print 'Predator action: %s' % action

    prey_by_distance = sorted(
      self.world.animats,
      key=lambda animat: \
        euclidean(self.world.predator.coords, animat.coords)
    )

    chosen_prey = None

    for animat in prey_by_distance:
      relative_color = abs(animat.color - BKGD_COLOR)

      if euclidean(self.world.predator.coords, prey_by_distance[0].coords) > DIST_MAX:
        continue

      elif any(rc >= COLOR_THRESHOLD for rc in relative_color):
        chosen_prey = animat
        break

    print 'Chosen prey: %s' % chosen_prey

    if chosen_prey is not None:

      print 'Predator coordinates: %s' % self.world.predator.coords

      vector  = chosen_prey.coords - self.world.predator.coords
      unitvec = vector / np.linalg.norm(vector)
      self.world.predator.coords += unitvec

      print 'Vector to prey: %s' % vector
      print 'Move one unit vector: %s' % unitvec
      print 'New coordinates: %s' % self.world.predator.coords

      dist = euclidean(self.world.predator.coords, chosen_prey.coords)

      print 'Distance to prey: %s' % dist

      if euclidean(self.world.predator.coords, chosen_prey.coords) <= DIST_THRESHOLD:

        if action == 'EAT':
          self.world.predator.eat(self.world.t, chosen_prey)
          self.world.predator.life += LIFE_UNIT

        elif action == 'REJECT':
          self.world.predator.reject(self.world.t, chosen_prey)

    self.world.predator.life -= LIFE_UNIT

    for p in self.world.animats:
      p.step(self.world.t)

  def getReward(self):
    """
    Give a reward based on the result of the action.
    """

    reward = 0.0

    if self.world.predator.eaten is not None:

      if isinstance(self.world.predator.eaten, PredatorAnimat):
        print 'Predator tried to eat another predator.'
        reward = -0.5

      else:
        print 'Predator ate a %s.' % type(self.world.predator.eaten).__name__
        self.world.predator.life += 1.0

        if self.world.predator.eaten.poison > 0.0:
          print 'Predator was poisoned %f units of life!' % self.world.predator.eaten.poison
          self.world.predator.life -= 1.0 + self.world.predator.eaten.poison
          self.world.predator.make_history(
            self.world.t, poison=self.world.predator.eaten.poison
          )
          if self.world.predator.life <= 0.0:
            print 'Predator died of poison!'
            reward = -1.0
          else:
            reward = -self.world.predator.eaten.poison

        else:
          reward = 1.0

        self.world.predator.eaten.respawn(self.world.t)

    elif self.world.predator.rejected is not None:
      print 'Predator rejected a %s.' % type(self.world.predator.rejected).__name__

      if isinstance(self.world.predator.rejected, PredatorAnimat):
        reward = 0.5

    else:
      print 'Predator was too far away from any animat to perform an action.'

    if self.world.predator.life <= 0.0:
      if self.world.predator.eaten is None:
        print 'Predator starved to death!'
      self.world.predator.respawn(self.world.t)

    self.world.predator.eaten    = None
    self.world.predator.rejected = None

    return reward

################################################################################
### helper subroutines

def get_color(i, nstates):
  i %= nstates
  b = i % (COLOR_MAX + 1)
  i = (i - b) / (COLOR_MAX + 1)
  g = i % (COLOR_MAX + 1)
  r = (i - g) / (COLOR_MAX + 1)
  return np.array([r,g,b])

def table_print(table, nstates):
  print '\n'.join(
    str(get_color(i, nstates)) + str(a)
    for i, a in enumerate(np.array_split(table, nstates))
  )

################################################################################
### main

if __name__ == '__main__':

  world = WorldInteraction()

  predTable = ActionValueTable(
    PredatorInteraction.NSTATES,
    len(PredatorInteraction.ACTIONS)
  )
  predTable.initialize(0.)

  predLearner = Q(ALPHA, GAMMA)
  predLearner._setExplorer(EpsilonGreedyExplorer(EPSILON))
  predAgent = LearningAgent(predTable, predLearner)

  predEnv = PredatorEnvironment(world)
  predTask = PredatorTask(predEnv)
  predExp = Experiment(predTask, predAgent)

  try:
    for t in xrange(MAX_TIME):
      print 't = %d' % t 
      world.t = t
      predExp.doInteractions(1)
      predAgent.learn()
      print 'Colors vs. Q-table:'
      table_print(predTable._params, PredatorInteraction.NSTATES)
      print

  except KeyboardInterrupt:
    pass

  finally:
    print 'Background: %s' % BKGD_COLOR
    print 'Colors vs. Final Q-table:'
    table_print(predTable._params, PredatorInteraction.NSTATES)
    print

    counts = {'ate' : {}, 'poison' : 0, 'death' : 0, 'poisondeath' : 0, 'rejected' : {}}

    for history in world.predator.history.values():

      if history.has_key('ate'):
        if not counts['ate'].has_key(history['ate']):
          counts['ate'][history['ate']] = 0
        counts['ate'][history['ate']] += 1

      if history.has_key('poison'):
        counts['poison'] += 1

      if history.has_key('death') and not history.has_key('poison'):
        counts['death'] += 1

      if history.has_key('poison') and history.has_key('poison'):
        counts['poisondeath'] += 1

      if history.has_key('rejected'):
        if not counts['rejected'].has_key(history['rejected']):
          counts['rejected'][history['rejected']] = 0
        counts['rejected'][history['rejected']] += 1

    print 'Predator history:'
    print '  Ate: %d (%s)' % (
      sum(counts['ate'].values()),
      ', '.join('%s: %d' % (k, v) for k, v in counts['ate'].iteritems())
    )
    print '  Rejected: %d (%s)' % (
      sum(counts['rejected'].values()),
      ', '.join('%s: %d' % (k, v) for k, v in counts['rejected'].iteritems())
    )
    print '  Poisoned: %d (Died: %d)' % (counts['poison'], counts['poisondeath'])
    print '  Starved: %d' % counts['death']

    #print '  Ate:                 %d' % len([t for t in world.predator.history.keys() if world.predator.history[t].has_key('ate')])
    #print '  Poisoned:            %d' % len([t for t in world.predator.history.keys() if world.predator.history[t].has_key('poison')])
    #print '  Death by starvation: %d' % len([t for t in world.predator.history.keys() if world.predator.history[t].has_key('death')])
    #print '  Death by poison:     %d' % len([t for t in world.predator.history.keys() if world.predator.history[t].has_key('death') and world.predator.history[t].has_key('poison')])
    #print '  Rejected:            %d' % len([t for t in world.predator.history.keys() if world.predator.history[t].has_key('rejected')])

### end
################################################################################



