# J14e Lane Counting Logic Documentation

## Overview
This document provides comprehensive details about the vehicle counting logic implemented in J14e.py for traffic analysis. The system tracks vehicles across multiple detection lines and counts specific movement patterns.

## Line Coordinates
- **Line A**: (299, 128) to (521, 135) - Green
- **Line B**: (18, 179) to (465, 303) - Blue  
- **Line C**: (489, 293) to (509, 140) - Red
- **Line D**: (18, 156) to (239, 132) - Yellow
- **Line E**: (214, 197) to (291, 186) - Magenta
- **B Supporter**: (10, 152) to (61, 208) - Purple

## Vehicle Classes
- **Truck Classes**: Truck, Pickup Truck, Heavy Truck, Machinery Vehicle
- **Bus Classes**: Bus
- **Other Classes**: Car, SUV, Van, Motorcycle

## Counting Logic Table

| Rule | Count Type | First Line | Required Lines | Special Conditions | Frame Requirement | Description |
|------|------------|------------|----------------|-------------------|-------------------|-------------|
| 1 | AB | A | C + B | C ignored, wait for B | 1 frame | A first, then C (ignored), then B |
| 2 | AB | A | B | None | 1 frame | A first, then B (simple) |
| 3 | AD | A | D | C ignored if exists | 1 frame | A first, then D |
| 4 | BA | B | D + A | D ignored, wait for A | 1 frame | B first, then D (ignored), then A |
| 5 | BC | B | D + C | D ignored, wait for C | 1 frame | B first, then D (ignored), then C |
| 6 | CA | C | A | Center bbox check (50%) | 1 frame | C first, then A with center bbox |
| 7 | CD | C | B or E | None | 1 frame | C first, then B or E |
| 8 | DB | D | B | A ignored if in middle | 1 frame | D first, then B |
| 9 | DC | D | C | None | 1 frame | D first, then C |
| 10 | DC | D | A | None | 10 frames | D first, then A for 10+ frames |
| Special | BA | Any | D/B + A | Lower bbox check (75%) | 1 frame | Any history of D/B + A (lower bbox) |
| Additional | AB | A | E | None | 1 frame | A first, then E |
| Additional | BC | B | E | None | 1 frame | B first, then E |

## Detailed Rule Explanations

### Rule 1: A → C → B = AB
- **First Line**: A
- **Required**: C (ignored) + B
- **Logic**: Vehicle intersects A first, then C (which is ignored), then B
- **Example**: A → C → B ✅, A → B → C ❌

### Rule 2: A → B = AB (Simple)
- **First Line**: A
- **Required**: B
- **Logic**: Vehicle intersects A first, then B directly
- **Example**: A → B ✅, B → A ❌

### Rule 3: A → D = AD
- **First Line**: A
- **Required**: D
- **Logic**: Vehicle intersects A first, then D (C ignored if present)
- **Example**: A → D ✅, A → C → D ✅, D → A ❌

### Rule 4: B → D → A = BA
- **First Line**: B
- **Required**: D (ignored) + A
- **Logic**: Vehicle intersects B first, then D (ignored), then A
- **Example**: B → D → A ✅, B → A → D ❌

### Rule 5: B → D → C = BC
- **First Line**: B
- **Required**: D (ignored) + C
- **Logic**: Vehicle intersects B first, then D (ignored), then C
- **Example**: B → D → C ✅, B → C → D ❌

### Rule 6: C → A = CA (Center Bbox)
- **First Line**: C
- **Required**: A (center bbox check)
- **Logic**: Vehicle intersects C first, then A with center 50% of bbox
- **Special**: Only middle 50% of vehicle bbox must intersect line A
- **Example**: C → A (center bbox) ✅, C → A (edge bbox) ❌

### Rule 7: C → B/E = CD
- **First Line**: C
- **Required**: B or E
- **Logic**: Vehicle intersects C first, then either B or E
- **Example**: C → B ✅, C → E ✅, C → A ❌

### Rule 8: D → B = DB
- **First Line**: D
- **Required**: B
- **Logic**: Vehicle intersects D first, then B (A ignored if in middle)
- **Example**: D → B ✅, D → A → B ✅, B → D ❌

### Rule 9: D → C = DC
- **First Line**: D
- **Required**: C
- **Logic**: Vehicle intersects D first, then C
- **Example**: D → C ✅, C → D ❌

### Rule 10: D → A (10 frames) = DC
- **First Line**: D
- **Required**: A for 10+ frames
- **Logic**: Vehicle intersects D first, then A for at least 10 frames
- **Example**: D → A (10+ frames) ✅, D → A (5 frames) ❌

### Special Rule: Any D/B + A (Lower Bbox) = BA
- **First Line**: Any
- **Required**: D or B + A (lower bbox)
- **Logic**: Any history of D or B, then A with lower 25% of bbox
- **Special**: Only lower 25% of vehicle bbox must intersect line A
- **Example**: D → A (lower bbox) ✅, B → A (lower bbox) ✅

### Additional Rule: A → E = AB
- **First Line**: A
- **Required**: E
- **Logic**: Vehicle intersects A first, then E
- **Example**: A → E ✅, E → A ❌

### Additional Rule: B → E = BC
- **First Line**: B
- **Required**: E
- **Logic**: Vehicle intersects B first, then E
- **Example**: B → E ✅, E → B ❌

## Bbox Check Details

### Center Bbox (50% - for CA rule)
```python
center_bbox_top = top + bbox_height * 0.25
center_bbox_bottom = top + bbox_height * 0.75
```
- Only middle 50% of vehicle bbox must intersect line A

### Lower Bbox (25% - for Special BA rule)
```python
lower_bbox_top = top + bbox_height * 0.75
lower_bbox_bottom = bottom
```
- Only lower 25% of vehicle bbox must intersect line A

## Frame Requirements

### 1 Frame Rules
- Most counting rules require 1 frame of intersection
- Applied to: AB, AD, BA, BC, CA, CD, DB, DC (Rule 9)

### 10 Frame Rules
- DC (Rule 10) requires 10 frames of A intersection after D
- Applied to: DC (Rule 10)

## Priority Order
Rules are checked in order from 1 to 10, plus special rules. First matching rule is applied.

## Vehicle Type Handling
- **Trucks/Buses**: Require tracking history and frame counting
- **Other Vehicles**: Count immediately when conditions are met
- **One Count Per Vehicle**: Each ID can only be counted once per session

## Line Intersection Logic
- **B Supporter Line**: Has HIGH PRIORITY for first intersection only
- **Subsequent Intersections**: Record normally without duplicates
- **Line History**: Maintains list of intersected lines per vehicle ID
- **Position Tracking**: Stores center coordinates and frame numbers

## Examples of Valid Counting Sequences

### AB Counting
1. A → B ✅
2. A → C → B ✅
3. A → E ✅

### AD Counting
1. A → D ✅
2. A → C → D ✅

### BA Counting
1. B → D → A ✅
2. B → A (lower bbox) ✅
3. D → A (lower bbox) ✅

### BC Counting
1. B → D → C ✅
2. B → E ✅

### CA Counting
1. C → A (center bbox) ✅

### CD Counting
1. C → B ✅
2. C → E ✅

### DB Counting
1. D → B ✅
2. D → A → B ✅

### DC Counting
1. D → C ✅
2. D → A (10+ frames) ✅

## Notes
- All coordinates are in pixel format (x, y)
- Line intersections are checked using bounding box collision detection
- Vehicle tracking uses DeepSORT for ID persistence
- Frame counting ensures accurate movement pattern detection
- Special bbox checks prevent false positives from partial intersections
