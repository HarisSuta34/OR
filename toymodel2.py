import pulp

# -----------------------------------
# Updated Input Data
# -----------------------------------

# Courses: course -> (enrollment, time_slot, list of valid rooms)
courses = {
    "Math101": (30, "Mon_9_11", ["F1.10"]),
    "CS102": (20, "Mon_9_11", ["F1.11"]),
    "Eng103": (40, "Mon_11_1", ["F1.23", "F1.24"]),
}

# Room capacities
room_capacity = {
    "F1.10": 65,
    "F1.11": 65,
    "F1.23": 50,
    "F1.24": 160
}

# Function to check if two courses overlap
def overlap(course1, course2):
    return courses[course1][1] == courses[course2][1]

# -----------------------------------
# Model Setup
# -----------------------------------

model = pulp.LpProblem("Classroom_Assignment_Group_Model", pulp.LpMinimize)

# Decision variables: x[c][r] = 1 if course c is assigned to room r
x = pulp.LpVariable.dicts("x", ((c, r) for c in courses for r in courses[c][2]), cat="Binary")

# -----------------------------------
# Objective Function
# Minimize total unused seat-hours
# -----------------------------------

model += pulp.lpSum([
    (room_capacity[r] - courses[c][0]) * x[c, r]
    for c in courses for r in courses[c][2]
]), "Total_Unused_Seat_Hours"

# -----------------------------------
# Constraints
# -----------------------------------

# 1. Each course assigned to exactly one room
for c in courses:
    model += pulp.lpSum([x[c, r] for r in courses[c][2]]) == 1, f"AssignOnce_{c}"

# 2. Capacity constraint: Room must be big enough for enrollment
for c in courses:
    for r in courses[c][2]:
        model += courses[c][0] * x[c, r] <= room_capacity[r], f"Capacity_{c}_{r}"

# 3. No time overlap in the same room
for r in room_capacity:
    for c1 in courses:
        for c2 in courses:
            if c1 != c2 and overlap(c1, c2):
                if r in courses[c1][2] and r in courses[c2][2]:
                    model += x[c1, r] + x[c2, r] <= 1, f"NoOverlap_{c1}_{c2}_{r}"

# -----------------------------------
# Solve the Model
# -----------------------------------

model.solve()
print("Status:", pulp.LpStatus[model.status])

# -----------------------------------
# Output Results
# -----------------------------------

for c in courses:
    for r in courses[c][2]:
        if pulp.value(x[c, r]) == 1:
            print(f"Course {c} assigned to room {r}")

print("Total unused seat-hours:", pulp.value(model.objective))
