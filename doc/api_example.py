from lap_tracker.detection import PeakDetector
from lap_tracker.tracker import Tracker

peaks_detector = PeakDetector('my_awesome_ome_file.tif')
peaks_detector.run(paralell=True)

tracker = Tracker(peaks_detector.peaks)
tracker.run()

tracker.show()

print(tracker.traj)
