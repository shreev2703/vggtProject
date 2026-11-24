from pathlib import Path

from vggt_bench.clip import ffmpeg_args, speedup_for


def test_speedup_targets_the_clip_length():
    assert speedup_for(80.0, target=8.0) == 10.0


def test_speedup_never_slows_a_short_clip():
    assert speedup_for(3.0, target=8.0) == 1.0


def test_ffmpeg_args_are_silent_web_h264():
    args = ffmpeg_args(Path("in.mov"), Path("out.mp4"), speed=5.0, width=640)
    assert "-an" in args
    assert "setpts=PTS/5.0000,fps=30,scale=640:-2" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[-1] == "out.mp4"
