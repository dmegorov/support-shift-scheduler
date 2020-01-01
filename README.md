# Balena support shift scheduler. Single day version

This is a fork of [Balena support shift scheduler](https://github.com/balena-io/support-shift-scheduler)
repository attempting to improve calculation time by taking an alternative approach to agent load spread optimization.

In brief, it takes previous days load running total for every agent as input and computes optimal schedule for a single
day. Agent loads from the resulting schedule can be added to the running totals thus providing input for the next day calculation.

This allows to reduce optimal schedule calculation time to several minutes on the provided sample input.

The amended `ortools_solver.py` script version takes JSON input of the same format as the original version
interpreting `agents/week_average_hours` values as agent's load running total in hours.
It also takes additional `--day <N>` parameter allowing to choose a day from the input arrays
(`N` is a zero-based index in the `agents/available_hours` arrays.) E.g.:

	python ../algo-core/ortools_solver.py --input support-shift-scheduler-input.json --day 2


## Rationale

Playing with some ideas around the model described in the related 
[blog post](https://www.balena.io/blog/the-unreasonable-effectiveness-of-algorithms-in-boosting-team-happiness/),
I noticed that the soft constraint number 3 is the only one that makes optimal solution lookup dependent on the input days span.
All other constraints can be calculated independently for each week day. 

_Note: the actual implementation also compensates late hour shifts which also introduces some dependency on the days span
but I'm leaving this alone for this experimental version. Support of this feature can be added later._

I had a feeling that eliminating days span dependency and calculating one day at a time could help improving calculation time dramatically.
So I tried to eliminate the dependency.

The constraint under question is responsible for spreading agent loads evenly over the course of a week.
How else can we ensure fair load distribution? In essence, fair load distribution means all agents would
spend equal time on duty over a long enough period of time (condition A).
Ideally, this period should be as short as possible (condition B).

A simple way to satisfy the conditions A and B is to

1. Generate schedules for the shortest periods possible.
	* A single day is the shortest period possible in our model. This would result in M = 1.

2. Assign shifts to the agents with least total load first.
	* In terms of our model, this would result in a soft constraint

		S(x) = k * SUM(t)[ SUM(n)[ d(nt) != 0 ? a(n) - MIN{a(1), ..., a(N)} : 0 ] ]

	  where a(n) is agent n load running total in hours.

This looks very similar to the soft constraint 4. Moreover, the forth constraint actually represents
a part of cost already presented in S(x) so we can simply replace S(4) by the new S(x).

The only open question remains is the coefficient k value. I believe, the penalty for uneven load distribution
should be at least as that for non-preferred hours on duty. So I stick with 8 initially. However, the final
value should be figured out by experiments as it was done for all other weight coefficients.


Technically, we end up with the original model but:

- without the soft constraint 3,
- domain count reduced by 1 (since M = 1),
- the soft constraint 4 changed its weight coefficient value and a(n) values having different meaning.


## Solver script amendments

In the initial script I made the following changes:

- Removed all solver variables, domains and constraints related to the soft constraint 3.
- Removed the day dimension from data frames involved into solver calculations.
- Extracted weight coefficients for all four soft constraints so one could easily experiment with various combinations.
  The respective variables can be found at the beginning of the `generate_schedule_with_ortools` function.
- Introduced a new input parameter `--day <N>` allowing to select a day from the initial input JSON file.


## Open issues and next steps

This implementation does not take into account previous day's late working hours for an agent.
Yet, this is not a big issue as these hours can be supplied as additional input data.

We also have to account for possible agent squad size changes. The team would most likely grow.
There is also a theoretical possibility to shrink. I did not think this thoroughly but it seems
to me such changes can be covered by specially crafted input load total values. The same for
holidays and other long leaves.
