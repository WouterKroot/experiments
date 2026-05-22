from psychopy import visual, core, event
import numpy as np 
# === Window setup ===
win = visual.Window(
    size=[800, 600],
    color=[0, 0, 0],
    units="pix",
    fullscr=False
)

# === Contrast values to inspect ===
# contrast_values = [0.001, 0.01, 0.013, 0.015, 0.018, 0.02, 0.025,
#                    0.03, 0.04, 0.05, 0.1, 0.2, 0.4,
#                    0.6, 0.8, 1.0, 1.2, 1.3, 1.5]
contrast_values = np.logspace(-3, 0, 20)
print()

# === Instruction text ===
instruction = visual.TextStim(
    win,
    text="Press SPACE to cycle through contrasts, ESC to quit.",
    pos=(0, 200),
    color="white"
)

# === Text to display current contrast ===
contrast_text = visual.TextStim(
    win,
    text="",
    pos=(0, -100),
    color="white",
    height=24
)

# === Main display loop ===
i = 0
while True:
    current_contrast = contrast_values[i]
    
    # Draw instruction
    instruction.draw()
    
    # Draw test line (black line, modulated by contrast)
    test_line = visual.Line(
        win=win,
        start=(-100, 0),
        end=(100, 0),
        lineColor="black",
        contrast=current_contrast,
        lineWidth=6
    )
    test_line.draw()
    
    # Draw contrast text
    contrast_text.text = f"Contrast: {current_contrast:.3f}"
    contrast_text.draw()
    
    # Flip window
    win.flip()
    
    # Wait for key input
    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        break
    elif "space" in keys:
        i = (i + 1) % len(contrast_values)

win.close()
core.quit()


# #%%
# import numpy as np

# def bounded_multipliers(baseline, multipliers):
#     """
#     Given a baseline contrast and a list of multiplicative factors,
#     return the scaled contrasts, clipped at 1.0.
#     """
#     print(baseline)
#     print(multipliers)
#     scaled = baseline * np.array(multipliers)
#     print(scaled)
#     scaled = np.minimum(scaled, 1.0)  # clip at max contrast
#     print(scaled)
#     return scaled

# # Example
# baseline = 0.015  # from staircase
# multipliers = [1, 1.5, 3, 8, 16, 32, 64]

# contrasts = bounded_multipliers(baseline, multipliers)
# print(contrasts)

# # %%
# import numpy as np

# def contrast_steps_log(baseline, max_val=1.0):
#     """
#     Generate contrast steps starting from baseline.
#     - The first few steps are fixed ratios: 1, 1.5, 3
#     - Remaining steps are log-spaced up to max_val (1.0)
#     """
#     # Define first steps
#     first_ratios = np.array([1, 1.5, 3])
#     first_steps = baseline * first_ratios
    
#     # Remove any first steps that exceed max_val
#     first_steps = first_steps[first_steps <= max_val]
    
#     # Remaining steps: logspace from last first step to max_val
#     if first_steps[-1] < max_val:
#         num_remaining = 7 - len(first_steps)  # total desired steps = 7
#         remaining_steps = np.logspace(
#             np.log10(first_steps[-1]),
#             np.log10(max_val),
#             num=num_remaining + 1,  # +1 to include max_val
#         )[1:]  # skip first point (already included)
#         contrasts = np.concatenate([first_steps, remaining_steps])
#     else:
#         contrasts = first_steps
    
#     return contrasts

# # Example usage:
# baseline = 0.02
# contrasts = contrast_steps_log(baseline)
# print(contrasts)
