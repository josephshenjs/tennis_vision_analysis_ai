from pathlib import Path
ROOT = Path(__file__).resolve().parent
BALL_SOURCE = ROOT/'RacketVision/source/BallTrack'
RACKET_SOURCE = ROOT/'RacketVision/source/RacketPose'
BALL_CHECKPOINT = BALL_SOURCE/'checkpoints/balltrack_best.pth'
BOUNCE_CHECKPOINT = ROOT/'models/bounce/ctb_regr_bounce.cbm'
RACKET_PYTHON = ROOT/'runtime/python.exe'
