#%%
from psychopy import visual, event, core
import json
import math

#  Setup window 
win = visual.Window([1000, 800], color='grey', units='pix', fullscr=False)

# Lines
lines = []
selected_index = None
dragging = False
mouse = event.Mouse(win=win)

def create_line(pos=(0, 0), angle=90, length=100):
    end1 = (pos[0] - math.cos(math.radians(angle)) * length / 2,
            pos[1] - math.sin(math.radians(angle)) * length / 2)
    end2 = (pos[0] + math.cos(math.radians(angle)) * length / 2,
            pos[1] + math.sin(math.radians(angle)) * length / 2)
    line = visual.Line(win, start=end1, end=end2, lineColor='white', lineWidth=3)
    return {
        'pos': pos,
        'angle': angle,
        'length': length,
        'line_obj': line
    }

def update_line(line_dict):
    angle = line_dict['angle']
    length = line_dict['length']
    pos = line_dict['pos']
    start = (pos[0] - math.cos(math.radians(angle)) * length / 2,
             pos[1] - math.sin(math.radians(angle)) * length / 2)
    end = (pos[0] + math.cos(math.radians(angle)) * length / 2,
           pos[1] + math.sin(math.radians(angle)) * length / 2)
    line_dict['line_obj'].start = start
    line_dict['line_obj'].end = end

def save_lines(filename='line_config.json'):
    out = []
    for l in lines:
        pos = l['pos']
        # Convert pos from numpy array to list if needed
        if hasattr(pos, 'tolist'):
            pos = pos.tolist()
        out.append({
            'pos': pos,
            'angle': l['angle'],
            'length': l['length']
        })
    with open(filename, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out)} lines to {filename}")

def is_mouse_near_line(mouse_pos, line_dict, tolerance=10):
    start = line_dict['line_obj'].start
    end = line_dict['line_obj'].end
    mx, my = mouse_pos
    sx, sy = start
    ex, ey = end
    # Line-point distance formula
    line_len = math.hypot(ex - sx, ey - sy)
    if line_len == 0:
        return False
    distance = abs((ey - sy) * mx - (ex - sx) * my + ex * sy - ey * sx) / line_len
    return distance < tolerance

def load_lines(filename='line_config.json'):
    global lines
    with open(filename, 'r') as f:
        data = json.load(f)
    lines = []
    for item in data:
        pos = tuple(item['pos'])  # convert list back to tuple
        angle = item['angle']
        length = item['length']
        line = create_line(pos=pos, angle=angle, length=length)
        lines.append(line)
    print(f"Loaded {len(lines)} lines from {filename}")
    
# === Main ===
load_lines('line_config.json')

while True:
    win.flip()
    for l in lines:
        l['line_obj'].draw()

    keys = event.getKeys()
    mouse_pos = mouse.getPos()

    # Handle key presses
    if 'q' in keys:
        save_lines()
        break
    if 'a' in keys:
        new_line = create_line(pos=mouse_pos)
        lines.append(new_line)
        selected_index = len(lines) - 1
    if 's' in keys:
        save_lines()

    if selected_index is not None:
        sel_line = lines[selected_index]
        # Adjust angle
        if 'left' in keys:
            sel_line['angle'] += 5
        if 'right' in keys:
            sel_line['angle'] -= 5
        # Adjust length
        if 'up' in keys:
            sel_line['length'] += 10
        if 'down' in keys:
            sel_line['length'] = max(10, sel_line['length'] - 10)
        update_line(sel_line)

    # Mouse click selects or drags a line
    if mouse.getPressed()[0]:  # Left click
        if not dragging:
            for i, l in enumerate(lines):
                if is_mouse_near_line(mouse_pos, l):
                    selected_index = i
                    dragging = True
                    break
    else:
        dragging = False

    if dragging and selected_index is not None:
        lines[selected_index]['pos'] = mouse.getPos()
        update_line(lines[selected_index])

    core.wait(0.01)

win.close()
core.quit()

