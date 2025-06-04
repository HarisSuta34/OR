import pandas as pd
import pulp

# --------------------------
# Load Data
# --------------------------

# Use your updated file with Day, Time, Time Slot columns
courses_df = pd.read_csv("CoursesWithTimeSlots.csv")

# Remove courses with missing required fields
courses_df = courses_df.dropna(subset=["Course Code", "Course Name", "Existing", "Time Slot"])
courses_df["Existing"] = courses_df["Existing"].astype(int)
courses_df = courses_df.reset_index(drop=True)

# --------------------------
# Filter out courses too big for any room
# --------------------------
rooms_df = pd.read_csv("Class Quotas  E-Campus (1).csv")
rooms_df = rooms_df.dropna(subset=["Name", "Teaching Capacity"])
rooms_df["Teaching Capacity"] = rooms_df["Teaching Capacity"].astype(int)
rooms_df = rooms_df.reset_index(drop=True)

max_capacity = rooms_df["Teaching Capacity"].max()
too_big = courses_df[courses_df["Existing"] > max_capacity]
courses_df = courses_df[courses_df["Existing"] <= max_capacity]
courses_df = courses_df.reset_index(drop=True)

# --------------------------
# Model Data Structures
# --------------------------

courses = []
for idx, row in courses_df.iterrows():
    course_id = f"{row['Course Code']}_{idx}"  # unique identifier for duplicate codes
    courses.append({
        "id": course_id,
        "code": row["Course Code"],
        "name": row["Course Name"],
        "enrollment": row["Existing"],
        "time": row["Time Slot"],
        "day": row["Day"] if "Day" in row else "",
        "slot_time": row["Time"] if "Time" in row else ""
    })

rooms = []
for idx, row in rooms_df.iterrows():
    rooms.append({
        "id": row["ID"],
        "name": row["Name"],
        "capacity": row["Teaching Capacity"]
    })

course_ids = [c["id"] for c in courses]
room_names = [r["name"] for r in rooms]
room_capacities = {r["name"]: r["capacity"] for r in rooms}

# Build a mapping from course id to its time slot and enrollment
course_time = {c["id"]: c["time"] for c in courses}
course_enroll = {c["id"]: c["enrollment"] for c in courses}
course_day = {c["id"]: c["day"] for c in courses}
course_slot_time = {c["id"]: c["slot_time"] for c in courses}

# --------------------------
# Set Up Optimization Model
# --------------------------

model = pulp.LpProblem("ClassroomAssignment", pulp.LpMinimize)

# Decision variables: x[course_id, room_name] = 1 if course assigned to that room
x = pulp.LpVariable.dicts("x", ((c, r) for c in course_ids for r in room_names), cat="Binary")

# Objective: Minimize total unused seat-hours
model += pulp.lpSum(
    (room_capacities[r] - course_enroll[c]) * x[c, r]
    for c in course_ids for r in room_names
), "Total_Unused_Seat_Hours"

# Constraint 1: Each course assigned to exactly one room
for c in course_ids:
    model += pulp.lpSum([x[c, r] for r in room_names]) == 1, f"AssignOnce_{c}"

# Constraint 2: Room capacity must be enough
for c in course_ids:
    for r in room_names:
        model += course_enroll[c] * x[c, r] <= room_capacities[r], f"Capacity_{c}_{r}"

# Constraint 3: No overlap in the same room at the same time
for r in room_names:
    for t in set(course_time.values()):
        overlapping_courses = [c for c in course_ids if course_time[c] == t]
        model += pulp.lpSum([x[c, r] for c in overlapping_courses]) <= 1, f"NoOverlap_{r}_{t}"

# --------------------------
# Solve
# --------------------------
print("Solving...")
status = model.solve(pulp.PULP_CBC_CMD(msg=False))
print("Status:", pulp.LpStatus[model.status])

# --------------------------
# Output Results
# --------------------------
output = []
for c in course_ids:
    for r in room_names:
        if pulp.value(x[c, r]) == 1:
            output.append({
                "Course Code": courses_df.loc[int(c.split('_')[-1]), "Course Code"],
                "Course Name": courses_df.loc[int(c.split('_')[-1]), "Course Name"],
                "Enrollment": courses_df.loc[int(c.split('_')[-1]), "Existing"],
                "Time Slot": courses_df.loc[int(c.split('_')[-1]), "Time Slot"],
                "Assigned Room": r,
                "Room Capacity": room_capacities[r],
                "Unused Seats": room_capacities[r] - courses_df.loc[int(c.split('_')[-1]), "Existing"]
            })

results_df = pd.DataFrame(output)
results_df.to_csv("assignment_results.csv", index=False)
print("Results saved to assignment_results.csv")
print("Total unused seat-hours:", pulp.value(model.objective))