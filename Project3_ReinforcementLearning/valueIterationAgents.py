# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Storing the previous value
        self.oldValues = self.values.copy()
        states = self.mdp.getStates()
        for i in range(self.iterations):
            for state in states:
                actions = self.mdp.getPossibleActions(state)  # Getting the possible actions
                if not self.mdp.isTerminal(state):
                    actionValue = -float('inf')
                    for action in actions:
                        qValue = self.computeQValueFromValues(state, action)  # Computing the Q value
                        actionValue = max(actionValue, qValue)  # Selecting max value between prev value and new value
                    self.values[state] = actionValue
            self.oldValues = self.values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Computing Q value from the formula in the algorithm
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += prob * (self.mdp.getReward(state, action, nextState)+self.discount * self.oldValues[nextState])
        return qValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if (self.mdp.isTerminal(state)):
            return None
        else:
            QValues = util.Counter()
            actions = self.mdp.getPossibleActions(state) # Getting the possible actions
            for action in actions:
                QValues[action] = self.computeQValueFromValues(state, action) #Computing Q value for given action

            return QValues.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        self.oldValues = self.values.copy()
        states = self.mdp.getStates()
        statesLen = len(states)
        for i in range(self.iterations):
            # Find state to update using mod, cyclic value iteration
            stateIndex = i % statesLen
            actions = self.mdp.getPossibleActions(states[stateIndex]) # Getting the possible actions
            if not self.mdp.isTerminal(states[stateIndex]):
                actionValue = -float('inf')
                for action in actions:
                    qValue = self.computeQValueFromValues(states[stateIndex], action)
                    actionValue = max(actionValue, qValue)
                self.values[states[stateIndex]] = actionValue
            self.oldValues = self.values.copy()

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pQueue = util.PriorityQueue()
        self.oldValues = self.values.copy()
        predecessors = {}
        # Compute predecessors
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state) # Getting the possible actions
            for action in actions:
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                if prob > 0:
                    if nextState in predecessors:
                        predecessors[nextState].add(state)
                    else:
                        predecessors[nextState] = {state}

        # Initialize priority Queue
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state) # Getting the possible actions
            actionValue = -float('inf')
            for action in actions:
              qValue = self.computeQValueFromValues(state, action)
              actionValue = max(actionValue, qValue)
            diff = abs(actionValue - self.values[state])
            pQueue.update(state, - diff)


        # Run iterations
        for i in range(self.iterations):
          if pQueue.isEmpty():
            break
          pop_state = pQueue.pop()
          if not self.mdp.isTerminal(pop_state):
            actionValue = -float('inf')
            actions = self.mdp.getPossibleActions(pop_state)
            for action in actions:
              qValue = self.computeQValueFromValues(pop_state, action)
              actionValue = max(actionValue, qValue)
            self.values[pop_state] = actionValue
            self.oldValues = self.values.copy()

          # Update priority Queue
          for predecessor in predecessors[pop_state]:
            if not self.mdp.isTerminal(predecessor):
              actionValue = -float('inf')
              for action in self.mdp.getPossibleActions(predecessor):
                qValue = self.computeQValueFromValues(predecessor, action)
                actionValue = max(actionValue, qValue)
              diff = abs(actionValue - self.values[predecessor])
              if diff > self.theta:
                pQueue.update(predecessor, -diff)

