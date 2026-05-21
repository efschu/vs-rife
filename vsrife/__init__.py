from __future__ import annotations

import math
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from threading import Lock
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vapoursynth as vs

from .__main__ import download_model

__version__ = "5.7.0"

os.environ["CI_BUILD"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

models = [
    "4.0",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "4.9",
    "4.10",
    "4.11",
    "4.12",
    "4.12.lite",
    "4.13",
    "4.13.lite",
    "4.14",
    "4.14.lite",
    "4.15",
    "4.15.lite",
    "4.16.lite",
    "4.17",
    "4.17.lite",
    "4.18",
    "4.19",
    "4.20",
    "4.21",
    "4.22",
    "4.22.lite",
    "4.23",
    "4.24",
    "4.25",
    "4.25.lite",
    "4.25.heavy",
    "4.26",
    "4.26.heavy",
]


@contextmanager
def redirect_stdout_to_stderr():
    old_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.close(old_stdout)


@dataclass
class GPUWorker:
    """Per-GPU state: model, streams, locks, constants, caches."""
    device: torch.device
    flownet: nn.Module | Callable
    encode: nn.Module | Callable | None
    dtype: torch.dtype
    inf_stream: torch.cuda.Stream
    inf_f2t_stream: torch.cuda.Stream
    inf_t2f_stream: torch.cuda.Stream
    inf_stream_lock: Lock
    inf_f2t_stream_lock: Lock
    inf_t2f_stream_lock: Lock
    enc_stream: torch.cuda.Stream | None = None
    enc_f2t_stream: torch.cuda.Stream | None = None
    enc_stream_lock: Lock | None = None
    enc_f2t_stream_lock: Lock | None = None
    tenFlow_div: torch.Tensor | None = None
    backwarp_tenGrid: torch.Tensor | None = None
    frame_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    encode_cache: dict[int, torch.Tensor] = field(default_factory=dict)


@redirect_stdout_to_stderr()
@torch.inference_mode()
def rife(
    clip: vs.VideoNode,
    device_index: int | list[int] = 0,
    model: str = "4.25",
    auto_download: bool = False,
    factor_num: int = 2,
    factor_den: int = 1,
    fps_num: int | None = None,
    fps_den: int | None = None,
    scale: float = 1.0,
    ensemble: bool = False,
    sc: bool = False,
    sc_threshold: float | None = 0.15,
    trt: bool = False,
    trt_static_shape: bool = True,
    trt_min_shape: list[int] = [128, 128],
    trt_opt_shape: list[int] = [1920, 1080],
    trt_max_shape: list[int] = [1920, 1080],
    trt_workspace_size: int = 0,
    trt_max_aux_streams: int | None = None,
    trt_optimization_level: int | None = None,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                     RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal(s) of the GPU(s). Pass a single int for one GPU,
                                     or a list of ints for multi-GPU round-robin distribution.
                                     With multiple GPUs, frame N is processed on GPU (N % num_gpus).
    :param model:                   Model to use.
    :param auto_download:           Automatically download the specified model if the file has not been downloaded.
    :param factor_num:              Numerator of factor for target frame rate.
    :param factor_den:              Denominator of factor for target frame rate.
                                     For example `factor_num=5, factor_den=2` will multiply the frame rate by 2.5.
    :param fps_num:                 Numerator of target frame rate.
    :param fps_den:                 Denominator of target frame rate.
                                     Override `factor_num` and `factor_den` if specified.
    :param scale:                   Control the process resolution for optical flow model. Try scale=0.5 for 4K video.
                                     Must be 0.25, 0.5, 1.0, 2.0, or 4.0.
    :param ensemble:                Smooth predictions in areas where the estimation is uncertain.
    :param sc:                      Avoid interpolating frames over scene changes.
    :param sc_threshold:            Threshold for scene change detection. Must be between 0.0 and 1.0.
                                     Leave the argument as None if the frames already have _SceneChangeNext property set.
    :param trt:                     Use TensorRT for high-performance inference.
                                     Not supported in '4.0' and '4.1' models.
    :param trt_static_shape:        Build with static or dynamic shapes.
    :param trt_min_shape:           Min size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_opt_shape:           Opt size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_max_shape:           Max size of dynamic shapes. Ignored if trt_static_shape=True.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_max_aux_streams:     Maximum number of auxiliary streams per inference stream that TRT is allowed to use
                                     to run kernels in parallel if the network contains ops that can run in parallel,
                                     with the cost of more memory usage. Set this to 0 for optimal memory usage.
                                     (default = using heuristics)
    :param trt_optimization_level:  Builder optimization level. Higher level allows TensorRT to spend more building time
                                     for more optimization options. Valid values include integers from 0 to the maximum
                                     optimization level, which is currently 5. (default is 3)
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                     first time. Note each engine is created for specific settings such as model
                                     path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("rife: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("rife: only RGBH and RGBS formats are supported")

    if clip.num_frames < 2:
        raise vs.Error("rife: clip's number of frames must be at least 2")

    if not torch.cuda.is_available():
        raise vs.Error("rife: CUDA is not available")

    if model not in models:
        raise vs.Error(f"rife: model must be one of {models}")

    if factor_num < 1:
        raise vs.Error("rife: factor_num must be at least 1")

    if factor_den < 1:
        raise vs.Error("rife: factor_den must be at least 1")

    if fps_num is not None and fps_num < 1:
        raise vs.Error("rife: fps_num must be at least 1")

    if fps_den is not None and fps_den < 1:
        raise vs.Error("rife: fps_den must be at least 1")

    if fps_num is not None and fps_den is not None and clip.fps == 0:
        raise vs.Error("rife: clip does not have a valid frame rate and hence fps_num and fps_den cannot be used")

    if scale not in [0.25, 0.5, 1.0, 2.0, 4.0]:
        raise vs.Error("rife: scale must be 0.25, 0.5, 1.0, 2.0, or 4.0")

    if not trt_static_shape:
        if not isinstance(trt_min_shape, list) or len(trt_min_shape) != 2:
            raise vs.Error("rife: trt_min_shape must be a list with 2 items")

        if any(trt_min_shape[i] < 1 for i in range(2)):
            raise vs.Error("rife: trt_min_shape must be at least 1")

        if not isinstance(trt_opt_shape, list) or len(trt_opt_shape) != 2:
            raise vs.Error("rife: trt_opt_shape must be a list with 2 items")

        if any(trt_opt_shape[i] < 1 for i in range(2)):
            raise vs.Error("rife: trt_opt_shape must be at least 1")

        if not isinstance(trt_max_shape, list) or len(trt_max_shape) != 2:
            raise vs.Error("rife: trt_max_shape must be a list with 2 items")

        if any(trt_max_shape[i] < 1 for i in range(2)):
            raise vs.Error("rife: trt_max_shape must be at least 1")

        if any(trt_min_shape[i] >= trt_max_shape[i] for i in range(2)):
            raise vs.Error("rife: trt_min_shape must be less than trt_max_shape")

    if os.path.getsize(os.path.join(model_dir, f"flownet_v{model}.pkl")) == 0:
        if auto_download:
            download_model(f"https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{model}.pkl")
        else:
            raise vs.Error(
                "rife: model file has not been downloaded. run `python -m vsrife` to download all models, or set "
                "`auto_download=True` to only download the specified model"
            )

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    if isinstance(device_index, int):
        devices = [torch.device("cuda", device_index)]
    elif isinstance(device_index, list):
        if len(device_index) < 1:
            raise vs.Error("rife: device_index list must contain at least one device")
        devices = [torch.device("cuda", idx) for idx in device_index]
    else:
        raise vs.Error("rife: device_index must be an int or a list of ints")

    for i, d in enumerate(devices):
        if d.index is None or d.index >= torch.cuda.device_count():
            raise vs.Error(f"rife: GPU {d.index} is not available (found {torch.cuda.device_count()} GPUs)")

    modulo = 32

    match model:
        case "4.0":
            from .IFNet_HDv3_v4_0 import IFNet

            Head = None
        case "4.1":
            from .IFNet_HDv3_v4_1 import IFNet

            Head = None
        case "4.2":
            from .IFNet_HDv3_v4_2 import IFNet

            Head = None
        case "4.3":
            from .IFNet_HDv3_v4_3 import IFNet

            Head = None
        case "4.4":
            from .IFNet_HDv3_v4_4 import IFNet

            Head = None
        case "4.5":
            from .IFNet_HDv3_v4_5 import IFNet

            Head = None
        case "4.6":
            from .IFNet_HDv3_v4_6 import IFNet

            Head = None
        case "4.7":
            from .IFNet_HDv3_v4_7 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
            encode_channel = 4
        case "4.8":
            from .IFNet_HDv3_v4_8 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
            encode_channel = 4
        case "4.9":
            from .IFNet_HDv3_v4_9 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))
            encode_channel = 4
        case "4.10":
            from .IFNet_HDv3_v4_10 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 8, 4, 2, 1),
                )
            encode_channel = 8
        case "4.11":
            from .IFNet_HDv3_v4_11 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 8, 4, 2, 1),
                )
            encode_channel = 8
        case "4.12":
            from .IFNet_HDv3_v4_12 import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 8, 4, 2, 1),
                )
            encode_channel = 8
        case "4.12.lite":
            from .IFNet_HDv3_v4_12_lite import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 4, 4, 2, 1),
                )
            encode_channel = 4
        case "4.13":
            from .IFNet_HDv3_v4_13 import Head, IFNet

            encode_channel = 8
        case "4.13.lite":
            from .IFNet_HDv3_v4_13_lite import IFNet

            with torch.device("meta"):
                Head = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.ConvTranspose2d(32, 4, 4, 2, 1),
                )
            encode_channel = 4
        case "4.14":
            from .IFNet_HDv3_v4_14 import Head, IFNet

            encode_channel = 8
        case "4.14.lite":
            from .IFNet_HDv3_v4_14_lite import Head, IFNet

            encode_channel = 8
        case "4.15":
            from .IFNet_HDv3_v4_15 import Head, IFNet

            encode_channel = 8
        case "4.15.lite":
            from .IFNet_HDv3_v4_15_lite import Head, IFNet

            encode_channel = 4
        case "4.16.lite":
            from .IFNet_HDv3_v4_16_lite import Head, IFNet

            encode_channel = 4
        case "4.17":
            from .IFNet_HDv3_v4_17 import Head, IFNet

            encode_channel = 8
        case "4.17.lite":
            from .IFNet_HDv3_v4_17_lite import Head, IFNet

            encode_channel = 4
        case "4.18":
            from .IFNet_HDv3_v4_18 import Head, IFNet

            encode_channel = 8
        case "4.19":
            from .IFNet_HDv3_v4_19 import Head, IFNet

            encode_channel = 8
        case "4.20":
            from .IFNet_HDv3_v4_20 import Head, IFNet

            encode_channel = 8
        case "4.21":
            from .IFNet_HDv3_v4_21 import Head, IFNet

            encode_channel = 8
        case "4.22":
            from .IFNet_HDv3_v4_22 import Head, IFNet

            encode_channel = 8
        case "4.22.lite":
            from .IFNet_HDv3_v4_22_lite import Head, IFNet

            encode_channel = 4
        case "4.23":
            from .IFNet_HDv3_v4_23 import Head, IFNet

            encode_channel = 8
        case "4.24":
            from .IFNet_HDv3_v4_24 import Head, IFNet

            encode_channel = 8
        case "4.25":
            from .IFNet_HDv3_v4_25 import Head, IFNet

            encode_channel = 4
            modulo = 64
        case "4.25.lite":
            from .IFNet_HDv3_v4_25_lite import Head, IFNet

            encode_channel = 4
            modulo = 128
        case "4.25.heavy":
            from .IFNet_HDv3_v4_25_heavy import Head, IFNet

            encode_channel = 4
            modulo = 64
        case "4.26":
            from .IFNet_HDv3_v4_26 import Head, IFNet

            encode_channel = 4
            modulo = 64
        case "4.26.heavy":
            from .IFNet_HDv3_v4_26_heavy import Head, IFNet

            encode_channel = 16
            modulo = 64

    model_name = f"flownet_v{model}.pkl"

    if fps_num is not None and fps_den is not None:
        factor = Fraction(fps_num, fps_den) / clip.fps
        factor_num, factor_den = factor.as_integer_ratio()

    if factor_num <= factor_den:
        raise vs.Error(
            "rife: target frame rate must be higher than source frame rate. consider using change_fps from "
            "https://github.com/Jaded-Encoding-Thaumaturgy/vs-tools if you want to reduce the frame rate"
        )

    w = clip.width
    h = clip.height
    tmp = max(modulo, int(modulo / scale))
    pw = math.ceil(w / tmp) * tmp
    ph = math.ceil(h / tmp) * tmp
    padding = (0, pw - w, 0, ph - h)
    need_pad = any(p > 0 for p in padding)

    if sc and sc_threshold is not None:
        clip = sc_detect(clip, sc_threshold)

    workers: list[GPUWorker] = []

    for dev_idx, device in enumerate(devices):
        if trt:
            import tensorrt
            import torch_tensorrt

            if trt_static_shape:
                dimensions = f"{pw}x{ph}"
            else:
                trt_min_s = list(trt_min_shape)
                trt_opt_s = list(trt_opt_shape)
                trt_max_s = list(trt_max_shape)
                for i in range(2):
                    trt_min_s[i] = math.ceil(trt_min_s[i] / tmp) * tmp
                    trt_opt_s[i] = math.ceil(trt_opt_s[i] / tmp) * tmp
                    trt_max_s[i] = math.ceil(trt_max_s[i] / tmp) * tmp

                dimensions = (
                    f"min-{trt_min_s[0]}x{trt_min_s[1]}"
                    f"_opt-{trt_opt_s[0]}x{trt_opt_s[1]}"
                    f"_max-{trt_max_s[0]}x{trt_max_s[1]}"
                )

            # Use first GPU to build engine (identical GPUs share engines)
            build_device = devices[0]
            build_dev_idx = 0

            gpu_suffix = f"_gpu{build_dev_idx}" if len(devices) > 1 else ""

            flownet_engine_path = os.path.join(
                os.path.realpath(trt_cache_dir),
                (
                    f"{model_name}"
                    + f"_{dimensions}"
                    + f"_{'fp16' if fp16 else 'fp32'}"
                    + f"_scale-{scale}"
                    + f"_ensemble-{ensemble}"
                    + gpu_suffix
                    + f"_{torch.cuda.get_device_name(build_device)}"
                    + f"_trt-{tensorrt.__version__}"
                    + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                    + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                    + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                    + ".ts"
                ),
            )

            encode_engine_path = flownet_engine_path + ".encode"

            if not os.path.isfile(flownet_engine_path) or (Head is not None and not os.path.isfile(encode_engine_path)):
                if sys.stdout is None:
                    sys.stdout = open(os.devnull, "w")

                flownet, encode = init_module(model_name, IFNet, scale, ensemble, build_device, dtype, Head)

                if trt_static_shape:
                    if encode is not None:
                        flownet_inputs = (
                            torch.zeros([1, 3, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([1, 3, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([1, 1, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([2], dtype=torch.float, device=build_device),
                            torch.zeros([1, 2, ph, pw], dtype=torch.float, device=build_device),
                            torch.zeros([1, encode_channel, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([1, encode_channel, ph, pw], dtype=dtype, device=build_device),
                        )

                        encode_inputs = (torch.zeros([1, 3, ph, pw], dtype=dtype, device=build_device),)
                    else:
                        flownet_inputs = (
                            torch.zeros([1, 3, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([1, 3, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([1, 1, ph, pw], dtype=dtype, device=build_device),
                            torch.zeros([2], dtype=torch.float, device=build_device),
                            torch.zeros([1, 2, ph, pw], dtype=torch.float, device=build_device),
                        )

                    flownet_dynamic_shapes = None
                    encode_dynamic_shapes = None
                else:
                    trt_min_s.reverse()
                    trt_opt_s.reverse()
                    trt_max_s.reverse()

                    if encode is not None:
                        flownet_inputs = (
                            torch.zeros([1, 3] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([1, 3] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([1, 1] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([2], dtype=torch.float, device=build_device),
                            torch.zeros([1, 2] + trt_opt_s, dtype=torch.float, device=build_device),
                            torch.zeros([1, encode_channel] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([1, encode_channel] + trt_opt_s, dtype=dtype, device=build_device),
                        )

                        encode_inputs = (torch.zeros([1, 3] + trt_opt_s, dtype=dtype, device=build_device),)
                    else:
                        flownet_inputs = (
                            torch.zeros([1, 3] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([1, 3] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([1, 1] + trt_opt_s, dtype=dtype, device=build_device),
                            torch.zeros([2], dtype=torch.float, device=build_device),
                            torch.zeros([1, 2] + trt_opt_s, dtype=torch.float, device=build_device),
                        )

                    _height = torch.export.Dim("height", min=trt_min_s[0] // tmp, max=trt_max_s[0] // tmp)
                    _width = torch.export.Dim("width", min=trt_min_s[1] // tmp, max=trt_max_s[1] // tmp)
                    dim_height = _height * tmp
                    dim_width = _width * tmp

                    if encode is not None:
                        flownet_dynamic_shapes = {
                            "img0": {2: dim_height, 3: dim_width},
                            "img1": {2: dim_height, 3: dim_width},
                            "timestep": {2: dim_height, 3: dim_width},
                            "tenFlow_div": {},
                            "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                            "f0": {2: dim_height, 3: dim_width},
                            "f1": {2: dim_height, 3: dim_width},
                        }

                        encode_dynamic_shapes = ({2: dim_height, 3: dim_width},)
                    else:
                        flownet_dynamic_shapes = {
                            "img0": {2: dim_height, 3: dim_width},
                            "img1": {2: dim_height, 3: dim_width},
                            "timestep": {2: dim_height, 3: dim_width},
                            "tenFlow_div": {},
                            "backwarp_tenGrid": {2: dim_height, 3: dim_width},
                        }

                flownet_program = torch.export.export(flownet, flownet_inputs, dynamic_shapes=flownet_dynamic_shapes)

                flownet = torch_tensorrt.dynamo.compile(
                    flownet_program,
                    flownet_inputs,
                    device=build_device,
                    num_avg_timing_iters=4,
                    workspace_size=trt_workspace_size,
                    min_block_size=1,
                    max_aux_streams=trt_max_aux_streams,
                    optimization_level=trt_optimization_level,
                    use_explicit_typing=True,
                )

                torch_tensorrt.save(flownet, flownet_engine_path, output_format="torchscript", inputs=flownet_inputs)

                if encode is not None:
                    encode_program = torch.export.export(encode, encode_inputs, dynamic_shapes=encode_dynamic_shapes)

                    encode = torch_tensorrt.dynamo.compile(
                        encode_program,
                        encode_inputs,
                        device=build_device,
                        num_avg_timing_iters=4,
                        workspace_size=trt_workspace_size,
                        min_block_size=1,
                        max_aux_streams=trt_max_aux_streams,
                        optimization_level=trt_optimization_level,
                        use_explicit_typing=True,
                    )

                    torch_tensorrt.save(encode, encode_engine_path, output_format="torchscript", inputs=encode_inputs)

            flownet = torch.jit.load(flownet_engine_path, map_location=device).eval()
            if Head is not None:
                encode = torch.jit.load(encode_engine_path, map_location=device).eval()
        else:
            flownet, encode = init_module(model_name, IFNet, scale, ensemble, device, dtype, Head)

        inf_stream = torch.cuda.Stream(device)
        inf_f2t_stream = torch.cuda.Stream(device)
        inf_t2f_stream = torch.cuda.Stream(device)

        inf_stream_lock = Lock()
        inf_f2t_stream_lock = Lock()
        inf_t2f_stream_lock = Lock()

        enc_stream = None
        enc_f2t_stream = None
        enc_stream_lock = None
        enc_f2t_stream_lock = None

        if Head is not None:
            enc_stream = torch.cuda.Stream(device)
            enc_f2t_stream = torch.cuda.Stream(device)
            enc_stream_lock = Lock()
            enc_f2t_stream_lock = Lock()

        tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=device)

        tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float, device=device)
        tenHorizontal = tenHorizontal.view(1, 1, 1, pw).expand(-1, -1, ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float, device=device)
        tenVertical = tenVertical.view(1, 1, ph, 1).expand(-1, -1, -1, pw)
        backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

        workers.append(GPUWorker(
            device=device,
            flownet=flownet,
            encode=encode,
            dtype=dtype,
            inf_stream=inf_stream,
            inf_f2t_stream=inf_f2t_stream,
            inf_t2f_stream=inf_t2f_stream,
            inf_stream_lock=inf_stream_lock,
            inf_f2t_stream_lock=inf_f2t_stream_lock,
            inf_t2f_stream_lock=inf_t2f_stream_lock,
            enc_stream=enc_stream,
            enc_f2t_stream=enc_f2t_stream,
            enc_stream_lock=enc_stream_lock,
            enc_f2t_stream_lock=enc_f2t_stream_lock,
            tenFlow_div=tenFlow_div,
            backwarp_tenGrid=backwarp_tenGrid,
        ))

    for w in workers:
        torch.cuda.current_stream(w.device).synchronize()

    num_workers = len(workers)

    @torch.inference_mode()
    def encoding(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        for worker in workers:
            with worker.enc_f2t_stream_lock, torch.cuda.stream(worker.enc_f2t_stream):
                img = frame_to_tensor(f, worker.device)

                if need_pad:
                    img = F.pad(img, padding)

                worker.enc_f2t_stream.synchronize()

                worker.frame_cache[n] = img

            with worker.enc_stream_lock, torch.cuda.stream(worker.enc_stream):
                output = worker.encode(img)

                worker.enc_stream.synchronize()

                worker.encode_cache[n] = output

        return f

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        worker = workers[n % num_workers]

        with worker.inf_f2t_stream_lock, torch.cuda.stream(worker.inf_f2t_stream):
            if worker.encode is not None:
                real_n = n * factor_den // factor_num
                real_n_next = min(real_n + 1, clip.num_frames - 1)

                cache_to_delete = real_n - 10

                if cache_to_delete >= 0:
                    if cache_to_delete in worker.frame_cache:
                        del worker.frame_cache[cache_to_delete]

                    if cache_to_delete in worker.encode_cache:
                        del worker.encode_cache[cache_to_delete]

            t = n * factor_den % factor_num / factor_num

            if t == 0 or (sc and f[0].props.get("_SceneChangeNext")):
                return f[0]

            if worker.encode is not None:
                if real_n in worker.frame_cache:
                    img0 = worker.frame_cache[real_n]
                else:
                    img0 = frame_to_tensor(f[0], worker.device)
                    if need_pad:
                        img0 = F.pad(img0, padding)

                if real_n_next in worker.frame_cache:
                    img1 = worker.frame_cache[real_n_next]
                else:
                    img1 = frame_to_tensor(f[1], worker.device)
                    if need_pad:
                        img1 = F.pad(img1, padding)

                if real_n in worker.encode_cache:
                    f0 = worker.encode_cache[real_n]
                else:
                    f0 = worker.encode(img0)

                if real_n_next in worker.encode_cache:
                    f1 = worker.encode_cache[real_n_next]
                else:
                    f1 = worker.encode(img1)
            else:
                img0 = frame_to_tensor(f[0], worker.device)
                img1 = frame_to_tensor(f[1], worker.device)
                if need_pad:
                    img0 = F.pad(img0, padding)
                    img1 = F.pad(img1, padding)

            timestep = torch.full([1, 1, ph, pw], t, dtype=worker.dtype, device=worker.device)

            worker.inf_f2t_stream.synchronize()

        with worker.inf_stream_lock, torch.cuda.stream(worker.inf_stream):
            if worker.encode is not None:
                output = worker.flownet(img0, img1, timestep, worker.tenFlow_div, worker.backwarp_tenGrid, f0, f1)
            else:
                output = worker.flownet(img0, img1, timestep, worker.tenFlow_div, worker.backwarp_tenGrid)

            worker.inf_stream.synchronize()

        with worker.inf_t2f_stream_lock, torch.cuda.stream(worker.inf_t2f_stream):
            if need_pad:
                output = output[:, :, :h, :w]

            return tensor_to_frame(output, f[0].copy(), worker.inf_t2f_stream)

    if Head is not None:
        encoded = clip.std.FrameEval(lambda n: clip.std.ModifyFrame(clip, encoding), clip_src=clip)
    else:
        encoded = clip

    clip0 = vs.core.std.Interleave([encoded] * factor_num)
    clip1 = encoded.std.DuplicateFrames(encoded.num_frames - 1)[1:]
    clip1 = vs.core.std.Interleave([clip1] * factor_num)
    if factor_den > 1:
        clip0 = clip0[::factor_den]
        clip1 = clip1[::factor_den]

    return clip0.std.FrameEval(lambda n: clip0.std.ModifyFrame([clip0, clip1], inference), clip_src=[clip0, clip1])


def init_module(
    model_name: str,
    IFNet: nn.Module,
    scale: float,
    ensemble: bool,
    device: torch.device,
    dtype: torch.dtype,
    Head: nn.Module | nn.Sequential | None,
) -> tuple[nn.Module, nn.Module | None]:
    state_dict = torch.load(os.path.join(model_dir, model_name), map_location="cpu", mmap=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "module." in k}

    with torch.device("meta"):
        flownet = IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False, assign=True)
    flownet.eval().to(device, dtype)

    if Head is not None:
        encode_state_dict = {k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k}

        if isinstance(Head, nn.Sequential):
            encode = Head
        else:
            with torch.device("meta"):
                encode = Head()
        encode.load_state_dict(encode_state_dict, assign=True)
        encode.eval().to(device, dtype)

        return flownet, encode

    return flownet, None


def sc_detect(clip: vs.VideoNode, threshold: float) -> vs.VideoNode:
    sc_clip = clip.resize.Bicubic(format=vs.GRAY8, matrix_s="709")

    sc_next = sc_clip[1:] + sc_clip[-1]
    sc_next = sc_next.std.PlaneStats(sc_clip)

    def set_sc_props(n, f):
        fout = f[0].copy()
        fout.props['_SceneChangeNext'] = int(threshold < f[1].props.get('PlaneStatsDiff', 0.0))
        return fout

    return clip.std.ModifyFrame(clips=[clip, sc_next], selector=set_sc_props)


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    ).unsqueeze(0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.squeeze(0).detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame
