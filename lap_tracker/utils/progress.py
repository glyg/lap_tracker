import sys


def pprogress(percent):
    """
    Print a progress bar
    percent = -1 to end and remove the progress bar
    """

    percent = int(percent)

    # If process is finished
    if percent == -1:
        sys.stdout.write("\r" + " " * 80 + "\r\r")
        sys.stdout.flush()
        return

    # Size of the progress bar
    size = 50

    # Compute current progress
    progress = (percent + 1) * size / 100

    # Build progress bar
    bar = "["
    for i in range(progress - 1):
        bar += "="
    bar += ">"
    for i in range(size - progress):
        bar += " "
    bar += "]"

    # Write progress bar
    sys.stdout.write("\r%d%% " % (percent + 1) + bar)
    sys.stdout.flush()
