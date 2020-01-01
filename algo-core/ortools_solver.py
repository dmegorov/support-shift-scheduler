# /*
# * Copyright 2019 Balena Ltd.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *    http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

import sys
import json
from datetime import timedelta
from dateutil.parser import parse as dateparse
import argparse
import jsonschema
from ortools.sat.python import cp_model
import math
import pandas as pd


def hours2range(week_hours):
    """ Convert per-hour availability flags into ranges format. """
    week_ranges = []

    for day_hours in week_hours:

        day_ranges = []
        start = None

        for i, value in enumerate(day_hours):

            # Start of new range:
            if start is None and value != 0:
                start = i
                continue

            # End of range:
            # (A range will end if either the current slot is unavailable
            # (value 0) or if the current slot is the last one.)
            if start is not None:
                if value == 0:  # Unavailable
                    day_ranges.append([start, i])
                    start = None
                elif i == end_hour - 1:  # Last slot
                    day_ranges.append([start, end_hour])
                else:
                    continue

        week_ranges.append(day_ranges)
    return week_ranges


def setup_dataframes():
    """ Set up dataframes for agents (df_a) and night shift info (df_n). """
    global min_load_total

    min_load_total = 100  # This will form a baseline for agent history.
    agents = input_json["agents"]

    df_a = pd.DataFrame(
        data=None,
        columns=[
            "Handle",
            "Email",
            "LoadTotals",
            "PrefIdealLength",
            "Hours",
            "HourRanges",
        ],
    )

    df_n_indices = pd.MultiIndex.from_product(
        [[t for t in range(num_tracks)], [d for d in range(num_days)]],
        names=("Track", "Day"),
    )

    df_n = pd.DataFrame(
        data="", columns=list(range(19, 24)), index=df_n_indices
    )

    for agent in agents:
        agent_load_total = math.trunc(float(agent["weekAverageHours"]))
        min_load_total = min(min_load_total, agent_load_total)
        week_hours = agent["availableHours"]

        for (d, _) in enumerate(week_hours):

            # Set availability to 0 outside balena support hours:
            for i in range(start_hour):
                week_hours[d][i] = 0
            for i in range(end_hour, num_slots):
                week_hours[d][i] = 0

            # Fill df_n dataframe with night shifts:
            indices_3 = [i for i, x in enumerate(week_hours[d]) if x == 3]

            if len(indices_3) > 0:

                start = indices_3[0]
                end = indices_3[-1] + 1

                if len(indices_3) == 5:
                    if list(df_n.loc[(0, d)]) == ["", "", "", "", ""]:
                        t = 0
                    else:
                        t = 1

                    for s in indices_3:
                        df_n.loc[(t, d), s] = agent["handle"]

                else:  # Always have the half-shifts in track t=1:
                    for s in indices_3:
                        df_n.loc[(1, d), s] = agent["handle"]

                # Reset agent preference to 2 for duration of night shift:
                week_hours[d][start:end] = [2 for i in range(start, end)]

                # Give agent a break until 15:00 the next day if he/she was
                # on night shift:
                if d != 4:
                    week_hours[d + 1][0:15] = [0 for i in range(15)]

        hour_ranges = hours2range(week_hours)

        df_a.loc[len(df_a)] = {
            "Handle": agent["handle"],
            "Email": agent["email"],
            "LoadTotals": agent_load_total,
            "PrefIdealLength": agent["idealShiftLength"],
            "Hours": week_hours,
            "HourRanges": hour_ranges,
        }

    # Hours: list of 5 lists, each of which has 24 items that mark the
    # availability of each hour (e.g.
    # [ [0,0,0,0,...,1,2,0,0], [0,0,0,0,...,1,2,0,0], [...], [...], [...] ])

    # HourRanges: list of 5 lists, each of the 5 lists has a number
    # of nested lists that mark the ranges that an agent is available to do
    # support (e.g. [ [[8,12], [16, 24]], [], [...], [...], [...])
    # NB: e.g. [8,12] indicates agent is available 8-12, NOT 8-13.

    df_a.set_index("Handle", inplace=True)
    return [df_a, df_n]


def get_unavailable_employees(day):
    """ Exclude employees with no availability for a given day. """
    dayNumber = day.weekday()
    unavailable = set()

    for handle in df_agents.index:
        if len(df_agents.loc[handle, "HourRanges"][dayNumber]) == 0:
            unavailable.add(handle)

    print("\nUnavailable employees on %s" % day)
    [print(e) for e in unavailable]
    return unavailable


def remove_agents_not_available_this_week():
    """ Agents not available at all this week are removed from the model. """
    print("")
    global df_agents

    for handle in df_agents.index:

        out = True

        for d in range(num_days):
            out = out and (handle in unavailable_employees)

        if out:
            df_agents.drop(index=handle, inplace=True)
            print(handle, "was removed for this week.")


def print_final_schedules(schedule_results):
    """ Print final schedule, validate output JSON, and write to file. """

    print(
        "\n%s shifts:"
        % schedule_results[0]["start_date"].strftime("%Y-%m-%d")
    )

    for (i, e) in enumerate(schedule_results[0]["shifts"]):
        print(e)

    output_json = []

    for epoch in schedule_results:
        # Substitute agent info from 'handle' to 'handle <email>'
        shifts = []

        for (name, start, end) in epoch["shifts"]:
            shifts.append(
                {
                    "agent": "%s <%s>" % (name, df_agents.loc[name, "Email"]),
                    "start": start,
                    "end": end,
                }
            )

        day_dict = {}
        day_dict["start_date"] = epoch["start_date"].strftime("%Y-%m-%d")
        day_dict["shifts"] = shifts
        output_json.append(day_dict)

    # JSON output format
    # {
    #   "start_date": YYYY-MM-DD # date is in YYYY-MM-DD format
    #   "shifts": [{
    #       "@agentHandle <agentEmail>": [ startHour, endHour ],
    #       '...'
    #   }]
    # }

    #    print(json.dumps(output_json, indent=4), file=sys.stdout)
    output_json_schema = json.load(
        open("../lib/schemas/support-shift-scheduler-output.schema.json")
    )

    try:
        jsonschema.validate(output_json, output_json_schema)
    except jsonschema.exceptions.ValidationError as err:
        print("Output JSON validation error", err)
        sys.exit(1)

    print("\nSuccessfully validated JSON output.")

    with open("support-shift-scheduler-output.json", "w") as outfile:
        outfile.write(json.dumps(output_json, indent=4))

    return output_json


def generate_schedule_with_ortools(day):
    """ Create and solve model with OR-Tools, producing final schedule. """
    global df_agents

    # Weight conefficients used by the model
    undesired_hours_weight = 8
    shorter_duration_weight = 3
    longer_duration_weight = 4
    agent_weight = 8
    handover_weight = 3


    # In this function, the following abbreviations are used:
    # t: track
    # h: Github handle
    # s: slot number

    model = cp_model.CpModel()

    # Constants / domains:
    d_hourCost = cp_model.Domain.FromValues([0, undesired_hours_weight])
    d_duration = cp_model.Domain.FromIntervals(
        [[0, 0], [min_duration, max_duration]]
    )

    # Create preference domains:
    d_prefs = pd.Series(data=None, index=df_agents.index)

    for h in df_agents.index:
        d_prefs.loc[h] = cp_model.Domain.FromIntervals(
            df_agents.loc[h, "HourRanges"][day]
        )

    # Indexed by track:
    v_t = pd.DataFrame(
        data=None, index=pd.RangeIndex(num_tracks), columns=["HandoverCost"]
    )

    # Indexed by track, handle:
    th_index_array = [[], []]

    for t in range(num_tracks):
        for h in df_agents.index:
            th_index_array[0].append(t)
            th_index_array[1].append(h)

    th_multi_index = pd.MultiIndex.from_arrays(
        th_index_array, names=("Track", "Handle")
    )

    v_th = pd.DataFrame(
        data=None,
        index=th_multi_index,
        columns=[
            "Start",
            "End",
            "Duration",
            "Interval",
            "IsAgentOn",
            "AgentCost",
            "IsDurationShorterThanIdeal",
            "DurationCost",
            "IsInPrefRange",
        ],
    )

    # Indexed by track, handle, slot:
    ths_index_array = [[], [], []]

    for t in range(num_tracks):
        for h in df_agents.index:
            for s in range(start_hour, end_hour):
                ths_index_array[0].append(t)
                ths_index_array[1].append(h)
                ths_index_array[2].append(s)

    ths_multi_index = pd.MultiIndex.from_arrays(
        ths_index_array, names=("Track", "Handle", "Slot")
    )

    v_ths = pd.DataFrame(
        data=None,
        index=ths_multi_index,
        columns=[
            "IsStartSmallerEqualHour",
            "IsEndGreaterThanHour",
            "IsHourCost",
            "HourCost",
        ],
    )

    # Fill dataframes with variables:

    # t:
    for t in range(num_tracks):
        v_t.loc[t, "HandoverCost"] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues([handover_weight * x for x in range(0, 8)]),
            "HandoverCost_%d" % t,
        )

    # th:
    print("")

    for t in range(num_tracks):
        for h in df_agents.index:
            when_on_night_shift = [
                19 + i
                for i, x in enumerate(df_nights.loc[(t, day)].to_list())
                if x == h
            ]

            if h in unavailable_employees:
                v_th.loc[(t, h), "Start"] = model.NewIntVar(
                    8, 8, "Start_%d_%s" % (t, h)
                )
                v_th.loc[(t, h), "End"] = model.NewIntVar(
                    8, 8, "End_%d_%s" % (t, h)
                )
                v_th.loc[(t, h), "Duration"] = model.NewIntVar(
                    0, 0, "Duration_%d_%s" % (t, h)
                )

            elif len(when_on_night_shift) > 0:
                start = when_on_night_shift[0]
                end = when_on_night_shift[-1] + 1
                duration = end - start

                v_th.loc[(t, h), "Start"] = model.NewIntVar(
                    start, start, "Start_%d_%s" % (t, h)
                )

                v_th.loc[(t, h), "End"] = model.NewIntVar(
                    end, end, "End_%d_%s" % (t, h)
                )

                v_th.loc[(t, h), "Duration"] = model.NewIntVar(
                    duration, duration, "Duration_%d_%s" % (t, h)
                )
                print(h + " on duty on night " + str(day + 1))

            else:
                v_th.loc[(t, h), "Start"] = model.NewIntVarFromDomain(
                    d_prefs.loc[h], "Start_%d_%s" % (t, h)
                )
                v_th.loc[(t, h), "End"] = model.NewIntVarFromDomain(
                    d_prefs.loc[h], "End_%d_%s" % (t, h)
                )
                v_th.loc[
                    (t, h), "Duration"
                ] = model.NewIntVarFromDomain(
                    d_duration, "Duration_%d_%s" % (t, h)
                )

            v_th.loc[(t, h), "Interval"] = model.NewIntervalVar(
                v_th.loc[(t, h), "Start"],
                v_th.loc[(t, h), "Duration"],
                v_th.loc[(t, h), "End"],
                "Interval_%d_%s" % (t, h),
            )

            v_th.loc[(t, h), "IsAgentOn"] = model.NewBoolVar(
                "IsAgentOn_%d_%s" % (t, h)
            )

            v_th.loc[(t, h), "AgentCost"] = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues([agent_weight * x for x in range(0, 65)]),
                "AgentCost_%d_%s" % (t, h),
            )

            v_th.loc[
                (t, h), "IsDurationShorterThanIdeal"
            ] = model.NewBoolVar(
                "IsDurationShorterThanIdeal_%d_%s" % (t, h)
            )

            duration_cost_list = set([shorter_duration_weight * x for x in range(0, 9)])
            duration_cost_list = list(
                duration_cost_list.union(set([longer_duration_weight * x for x in range(0, 9)]))
            )
            duration_cost_list.sort()

            v_th.loc[
                (t, h), "DurationCost"
            ] = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(duration_cost_list),
                "DurationCost_%d_%s" % (t, h),
            )

            v_th.loc[(t, h), "IsInPrefRange"] = [
                model.NewBoolVar("IsInPrefRange_%d_%s_%d" % (t, h, j))
                for (j, sec) in enumerate(df_agents.loc[h, "HourRanges"][day])
            ]

    # ths:
    for t in range(num_tracks):
        for h in df_agents.index:
            for s in range(start_hour, end_hour):
                v_ths.loc[
                    (t, h, s), "IsStartSmallerEqualHour"
                ] = model.NewBoolVar(
                    "IsStartSmallerEqualHour_%d_%s_%d" % (t, h, s)
                )

                v_ths.loc[
                    (t, h, s), "IsEndGreaterThanHour"
                ] = model.NewBoolVar(
                    "IsEndGreaterThanHour_%d_%s_%d" % (t, h, s)
                )

                v_ths.loc[(t, h, s), "IsHourCost"] = model.NewBoolVar(
                    "IsHourCost_%d_%s_%d" % (t, h, s)
                )

                v_ths.loc[
                    (t, h, s), "HourCost"
                ] = model.NewIntVarFromDomain(
                    d_hourCost, "HourCost_%d_%s_%d" % (t, h, s)
                )

    # Constraint: The sum of agents' shifts must equal work_hours:
    for t in range(num_tracks):
        model.Add(
            sum(v_th.loc[(t), "Duration"].values.tolist()) == work_hours
        )

    # Constraint: Agent shifts must not overlap with each other:
    for t in range(num_tracks):
        model.AddNoOverlap(v_th.loc[(t), "Interval"].values.tolist())

    # Constraint: Honour agent availability requirements - a shift
    # must start and end within an agent's availability hours:
    # NB: AddBoolOr works with just one boolean as well, in which case that
    # boolean has to be true.
    for t in range(num_tracks):
        for h in df_agents.index:
            if not (h in unavailable_employees):
                model.AddBoolOr(v_th.loc[(t, h), "IsInPrefRange"])

                for (j, sec) in enumerate(
                    df_agents.loc[h, "HourRanges"][day]
                ):
                    model.Add(
                        v_th.loc[(t, h), "Start"] >= sec[0]
                    ).OnlyEnforceIf(
                        v_th.loc[(t, h), "IsInPrefRange"][j]
                    )
                    model.Add(
                        v_th.loc[(t, h), "Start"]
                        + v_th.loc[(t, h), "Duration"]
                        <= sec[1]
                    ).OnlyEnforceIf(
                        v_th.loc[(t, h), "IsInPrefRange"][j]
                    )

    # Constraint: Ensure agent not scheduled for more than one track at a time:
    for h in df_agents.index:

        isAgentOn_list = []

        for t in range(num_tracks):
            isAgentOn_list.append(v_th.loc[(t, h), "IsAgentOn"].Not())

        model.AddBoolOr(isAgentOn_list)

    # Constraint: Add other cost terms:
    for t in range(num_tracks):
        # Add cost due to number of handovers:
        model.Add(
            v_t.loc[t, "HandoverCost"]
            == handover_weight
            * (sum(v_th.loc[t, "IsAgentOn"].values.tolist()) - 1)
        )

        for h in df_agents.index:
            # Put toggles in place reflecting whether agent was assigned:
            model.Add(v_th.loc[(t, h), "Duration"] != 0).OnlyEnforceIf(
                v_th.loc[(t, h), "IsAgentOn"]
            )
            model.Add(v_th.loc[(t, h), "Duration"] == 0).OnlyEnforceIf(
                v_th.loc[(t, h), "IsAgentOn"].Not()
            )

            # Add cost due to agent history:
            agent_cost = agent_weight * (
                df_agents.loc[h, "LoadTotals"] - min_load_total
            )

            model.Add(
                v_th.loc[(t, h), "AgentCost"] == agent_cost
            ).OnlyEnforceIf(v_th.loc[(t, h), "IsAgentOn"])
            model.Add(v_th.loc[(t, h), "AgentCost"] == 0).OnlyEnforceIf(
                v_th.loc[(t, h), "IsAgentOn"].Not()
            )

            # Add cost due to shift duration:
            model.Add(
                v_th.loc[(t, h), "Duration"]
                < df_agents.loc[h, "PrefIdealLength"]
            ).OnlyEnforceIf(
                v_th.loc[(t, h), "IsDurationShorterThanIdeal"]
            )
            model.Add(
                v_th.loc[(t, h), "Duration"]
                >= df_agents.loc[h, "PrefIdealLength"]
            ).OnlyEnforceIf(
                v_th.loc[(t, h), "IsDurationShorterThanIdeal"].Not()
            )

            # Cost for zero duration:
            model.Add(
                v_th.loc[(t, h), "DurationCost"] == 0
            ).OnlyEnforceIf(v_th.loc[(t, h), "IsAgentOn"].Not())

            # Cost for duration shorter than preference:
            model.Add(
                v_th.loc[(t, h), "DurationCost"]
                == shorter_duration_weight
                * (
                    df_agents.loc[h, "PrefIdealLength"]
                    - v_th.loc[(t, h), "Duration"]
                )
            ).OnlyEnforceIf(
                [
                    v_th.loc[(t, h), "IsAgentOn"],
                    v_th.loc[(t, h), "IsDurationShorterThanIdeal"],
                ]
            )

            # Cost for duration longer than preference:
            model.Add(
                v_th.loc[(t, h), "DurationCost"]
                == longer_duration_weight
                * (
                    v_th.loc[(t, h), "Duration"]
                    - df_agents.loc[h, "PrefIdealLength"]
                )
            ).OnlyEnforceIf(
                v_th.loc[(t, h), "IsDurationShorterThanIdeal"].Not()
            )

            # Add hour cost:
            for (s_count, s_cost) in enumerate(
                df_agents.loc[h, "Hours"][day][start_hour:end_hour]
            ):

                s = s_count + start_hour

                model.Add(v_th.loc[(t, h), "Start"] <= s).OnlyEnforceIf(
                    v_ths.loc[(t, h, s), "IsStartSmallerEqualHour"]
                )
                model.Add(v_th.loc[(t, h), "Start"] > s).OnlyEnforceIf(
                    v_ths.loc[
                        (t, h, s), "IsStartSmallerEqualHour"
                    ].Not()
                )

                model.Add(v_th.loc[(t, h), "End"] > s).OnlyEnforceIf(
                    v_ths.loc[(t, h, s), "IsEndGreaterThanHour"]
                )
                model.Add(v_th.loc[(t, h), "End"] <= s).OnlyEnforceIf(
                    v_ths.loc[(t, h, s), "IsEndGreaterThanHour"].Not()
                )

                model.AddBoolAnd(
                    [
                        v_ths.loc[(t, h, s), "IsStartSmallerEqualHour"],
                        v_ths.loc[(t, h, s), "IsEndGreaterThanHour"],
                    ]
                ).OnlyEnforceIf(v_ths.loc[(t, h, s), "IsHourCost"])

                model.AddBoolOr(
                    [
                        v_ths.loc[
                            (t, h, s), "IsStartSmallerEqualHour"
                        ].Not(),
                        v_ths.loc[
                            (t, h, s), "IsEndGreaterThanHour"
                        ].Not(),
                    ]
                ).OnlyEnforceIf(
                    v_ths.loc[(t, h, s), "IsHourCost"].Not()
                )

                model.Add(
                    v_ths.loc[(t, h, s), "HourCost"]
                    == undesired_hours_weight * (s_cost - 1)
                ).OnlyEnforceIf(v_ths.loc[(t, h, s), "IsHourCost"])

                model.Add(
                    v_ths.loc[(t, h, s), "HourCost"] == 0
                ).OnlyEnforceIf(
                    v_ths.loc[(t, h, s), "IsHourCost"].Not()
                )

    full_cost_list = (
        v_t["HandoverCost"].values.tolist()
        + v_th["AgentCost"].values.tolist()
        + v_th["DurationCost"].values.tolist()
        + v_ths["HourCost"].values.tolist()
    )

    model.Minimize(sum(full_cost_list))
    print(model.Validate())

    # Solve model:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_timeout
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    print(solver.StatusName(status))

    # Extract solution:
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Cannot create schedule")
        return

    else:
        print("\n---------------------")
        print("| OR-Tools schedule |")
        print("---------------------")

        print("\nSolution type: ", solver.StatusName(status))
        print("\nMinimized cost: ", solver.ObjectiveValue())
        print("After", solver.WallTime(), "seconds.")
        schedule_results = []

        day_dict = {}
        day_dict["start_date"] = days[day]
        day_dict["shifts"] = []

        for t in range(num_tracks):
            for h in df_agents.index:
                if solver.Value(v_th.loc[(t, h), "Duration"]) != 0:
                    day_dict["shifts"].append(
                        (
                            h,
                            solver.Value(v_th.loc[(t, h), "Start"]),
                            solver.Value(v_th.loc[(t, h), "End"]),
                        )
                    )

        schedule_results.append(day_dict)

        # Sort shifts by start times to improve output readability:
        for i in range(len(schedule_results)):
            shifts = schedule_results[i]["shifts"]
            sorted_shifts = sorted(shifts, key=lambda x: x[1])
            schedule_results[i]["shifts"] = sorted_shifts

        return print_final_schedules(schedule_results)


# MAIN CODE BLOCK

# Production (read input from command line):

sys.stderr.write("Command line args: %s\n" % sys.argv)
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", help="Scheduler input JSON file path", required=True
)
parser.add_argument(
    "-d", "--day", type=int, help="A zero-base day index in the input JSON file", required=False, default=0
)
args = parser.parse_args()
input_filename = args.input.strip()
selected_day = args.day

# Testing (define input directly):

# input_filename = 'support-shift-scheduler-input.json'

# Load and validate JSON input:

input_json = json.load(open(input_filename))
input_json_schema = json.load(
    open("../lib/schemas/support-shift-scheduler-input.schema.json")
)
try:
    jsonschema.validate(input_json, input_json_schema)
except jsonschema.exceptions.ValidationError as err:
    print("Input JSON validation error", err)
    sys.exit(1)

# Define variables from options:
scheduler_options = input_json["options"]
start_Monday = scheduler_options["startMondayDate"]
num_days = int(scheduler_options["numConsecutiveDays"])
num_tracks = int(scheduler_options["numSimultaneousTracks"])
start_hour = int(scheduler_options["supportStartHour"])
end_hour = int(scheduler_options["supportEndHour"])
min_duration = int(scheduler_options["shiftMinDuration"])
max_duration = int(scheduler_options["shiftMaxDuration"])
solver_timeout = int(scheduler_options["optimizationTimeout"])

# Other global variables:
work_hours = end_hour - start_hour
num_slots = 24

start_date = dateparse(start_Monday).date()
delta = timedelta(days=1)
days = [start_date]
selected_date = start_date + selected_day * delta

for d in range(1, num_days):
    days.append(days[d - 1] + delta)

[df_agents, df_nights] = setup_dataframes()

# Determine unavailable agents for each day:
unavailable_employees = get_unavailable_employees(selected_date)
'''unavailable_employees = []

for d in range(num_days):
    unavailable_employees.append(get_unavailable_employees(days[d]))'''

# Remove agents from the model who are not available at all this week:
remove_agents_not_available_this_week()

# Create schedule:
output_sched = generate_schedule_with_ortools(selected_day)
