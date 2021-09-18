# Reinforcement Learning Study Project

## Motivation

Data Science is dead, and machine learning has hit a point of diminishing
returns.  Plus, training machine learning models has stopped being new and
exciting for me.  I need to learn AI.  Not only because it's interesting, but
because that's what I should be doing if I don't want to get stuck in middle
management for the rest of my career.

## The Plan

How to build up to this?

1. Study reinforcement learning.  The [Sutton and
   Barto](http://incompleteideas.net/book/the-book-2nd.html) is the canonical
   reference.  I plan to read the most important chapters to get a sense of
   deep RL.
2. Smaller projects.  Start smaller by training a tabular policy on a small
   state space problem, e.g., snake, tetris.
3. Medium project.  Chess might be ideal for this since I already have a chess
   app and can train against stockfish.
3. Robotics.  I have this idea to build a fetch robot powered by a raspberry
   pi.  There are beginner friendly rc car kits for use with the pi, and I
   could train an agent to use the sensor info from one of those.

## Ideas

### Rethink Snake AI

I already have built a genetic algorithm to play snake
[here](https://github.com/jonathanlamar/snake-learning).  While this is
technically an RL algorithm, it does not utilize any notion of value and only
reacts to maximize immediate rewards.  I think it could be greatly improved
with a proper theoretical grounding.

### Chess Bot

I also built a chess program
[here](https://github.com/jonathanlamar/chess-app).  It also has an AI that is
based on alpha-beta pruning.  This is also technically an RL algorithm and even
optimizes for long term value.  However, I think I can experiment with deep
learning here.

### Fetch Robot

A self driving car.  Not an actual car, where the risks and rewards make the
problem much more challenging, but rather an RC car powered by a raspberry pi.
So this won't be able to support real time image segmentation on a video feed.
It may be possible to incorporate a webcam, but only for static images.

What is the objective of this fake roomba?  It should be a simple objective,
that is amenable to reinforcement learning.  Perhaps rig a small object with a
reflector or something and teach the roomba to go find the object and return it
to a specified goal.  This may be too advanced for me right now, but represents
a longer term goal that should be feasible.
