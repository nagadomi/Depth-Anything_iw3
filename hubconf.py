import torch
from os import path


def _load_state_dict(encoder, model_type=None):
    if model_type is None:
        if encoder in {"vits", "vitb", "vitl"}:
            file_name = f"depth_anything_{encoder}14.pth"
            url = f"https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/{file_name}?download=true"
            state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name,
                                                            weights_only=True, map_location=torch.device("cpu"))
            return state_dict
        elif encoder in {"v2_vits"}:
            file_name = f"depth_anything_{encoder}.pth"
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/{file_name}?download=true"
            state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name,
                                                            weights_only=True, map_location=torch.device("cpu"))
            return state_dict
        elif encoder in {"v2_vitb", "v2_vitl"}:
            file_name = f"depth_anything_{encoder}.pth"
            checkpoint_path = path.join(torch.hub.get_dir(), "checkpoints", file_name)
            if path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
                return state_dict
            else:
                raise RuntimeError(f"Please place the checkpoint file for cc-by-nc-4.0 yourself.\n{checkpoint_path}")
        else:
            raise ValueError(f"Unknown encoder {encoder}")
    else:
        file_name = None
        url = None
        patch = None
        if model_type == "hypersim":
            file_name = f"depth_anything_v2_metric_{model_type}_{encoder[3:]}.pth"
            if encoder == "v2_vits":
                url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/{file_name}?download=true"
            elif encoder == "v2_vitb":
                url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/{file_name}?download=true"
        elif model_type == "vkitti":
            file_name = f"depth_anything_v2_metric_{model_type}_{encoder[3:]}.pth"
            if encoder == "v2_vits":
                url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/{file_name}?download=true"
            elif encoder == "v2_vitb":
                url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/{file_name}?download=true"
        elif model_type == "distill_any_depth":
            if encoder == "v2_vits":
                file_name = "distill_any_depth_vits.safetensors"
                url = "https://huggingface.co/xingyang1/Distill-Any-Depth/resolve/main/small/model.safetensors?download=true"
            elif encoder == "v2_vitb":
                file_name = "distill_any_depth_vitb.safetensors"
            elif encoder == "v2_vitl":
                file_name = "distill_any_depth_vitl.safetensors"

                def _patch(state_dict):
                    state_dict_new = {}
                    for key in state_dict:
                        new_key = key.replace("backbone.blocks.0.", "pretrained.blocks.").replace("backbone.", "pretrained.")
                        state_dict_new[new_key] = state_dict[key]
                    return state_dict_new
                patch = _patch
            else:
                raise ValueError(f"Unknown encoder {model_type} {encoder}")
        elif model_type == "ultimate_dav2":
            if encoder == "v2_vits":
                file_name = "ultimate_dav2_vits_INet_quality.safetensors"
            elif encoder == "v2_vitb":
                file_name = "ultimate_dav2_vitB_INet_quality.safetensors"
            elif encoder == "v2_vits_ul":
                file_name = "ultimate_dav2_vits_INet_UL.safetensors"
            else:
                raise ValueError(f"Unknown encoder {model_type} {encoder}")
        else:
            raise ValueError(f"Unknown encoder {encoder}")

        if url is not None:
            if ".safetensors" in url:
                import safetensors.torch
                checkpoint_path = path.join(torch.hub.get_dir(), "checkpoints", file_name)
                if not path.exists(checkpoint_path):
                    torch.hub.download_url_to_file(url, checkpoint_path)
                state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
            else:
                state_dict = torch.hub.load_state_dict_from_url(url, file_name=file_name,
                                                                weights_only=True, map_location=torch.device("cpu"))
            if callable(patch):
                state_dict = patch(state_dict)
            return state_dict
        else:
            checkpoint_path = path.join(torch.hub.get_dir(), "checkpoints", file_name)
            if path.exists(checkpoint_path):
                if file_name.endswith(".safetensors"):
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
                else:
                    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
                if callable(patch):
                    state_dict = patch(state_dict)
                return state_dict
            else:
                raise RuntimeError(f"Please place the checkpoint file yourself.\n{checkpoint_path}")


def DepthAnything(encoder, model_type=None, localhub=True):
    from depth_anything.dpt import DPT_DINOv2
    assert encoder in {"vits", "vitb", "vitl", "v2_vits", "v2_vitb", "v2_vitl"}

    depth_anything = DPT_DINOv2(encoder=encoder, localhub=localhub)
    depth_anything.load_state_dict(_load_state_dict(encoder, model_type), strict=True)

    return depth_anything


def DistillAnyDepth(encoder, localhub=True):
    """ https://github.com/Westlake-AGI-Lab/Distill-Any-Depth
    """
    from depth_anything.dpt import DPT_DINOv2
    assert encoder in {"v2_vits", "v2_vitb", "v2_vitl"}

    depth_anything = DPT_DINOv2(encoder=encoder, localhub=localhub)
    depth_anything.load_state_dict(_load_state_dict(encoder, model_type="distill_any_depth"), strict=True)

    return depth_anything


def UltimateDAV2(encoder, localhub=True):
    """ https://github.com/nagadomi/nunif/discussions/727
    """
    from depth_anything.dpt import DPT_DINOv2
    assert encoder in {"v2_vits", "v2_vitb", "v2_vits_ul"}

    depth_anything = DPT_DINOv2(encoder=encoder, localhub=localhub)
    depth_anything.load_state_dict(_load_state_dict(encoder, model_type="ultimate_dav2"), strict=True)

    return depth_anything


def DepthAnythingMetricDepth(model_type="indoor", remove_prep=True):
    model = torch.hub.load(path.join(path.dirname(__file__), "metric_depth"),
                           "DepthAnythingMetricDepth",
                           model_type=model_type, remove_prep=remove_prep,
                           source="local")
    return model


def DepthAnythingMetricDepthV2(model_type="hypersim", localhub=True):
    from depth_anything.dpt import DPT_DINOv2
    model_type = "hypersim_l" if model_type == "hypersim" else model_type
    model_type = "vkitti_l" if model_type == "vkitti" else model_type
    assert model_type in {"hypersim_l", "hypersim_b", "hypersim_s",
                          "vkitti_l", "vkitti_b", "vkitti_s"}

    encoder = {"l": "v2_vitl", "b": "v2_vitb", "s": "v2_vits"}[model_type[-1]]
    if "hypersim" in model_type:
        model_type = "hypersim"
        max_depth = 20.0
    else:
        model_type = "vkitti"
        max_depth = 80.0

    depth_anything = DPT_DINOv2(encoder=encoder, localhub=localhub, max_depth=max_depth, metric_depth=True)
    depth_anything.load_state_dict(_load_state_dict(encoder, model_type), strict=True)

    return depth_anything


def transforms_cv2():
    from torchvision.transforms import Compose
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    import cv2

    class PackImage():
        def __call__(self, image):
            return {"image": image}

    class UnpackImage():
        def __call__(self, sample):
            return sample["image"]

    class BGRToRGBFloat():
        def __call__(self, sample):
            sample["image"] = cv2.cvtColor(sample["image"], cv2.COLOR_BGR2RGB) / 255.0
            return sample

    class ToTensor():
        def __call__(self, image):
            return torch.from_numpy(image)

    transform = Compose([
        PackImage(),
        BGRToRGBFloat(),
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        UnpackImage(),
        ToTensor(),
    ])
    return transform


def transforms_pil(lower_bound=518, ensure_multiple_of=14, normalize=True, normalize_mode="imagenet"):
    import torchvision.transforms.functional as TF
    from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode

    assert normalize_mode in {"imagenet", "center"}

    class ModConstraintLowerBoundResize():
        def __init__(self, lower_bound=518, ensure_multiple_of=14):
            self.lower_bound = lower_bound
            self.ensure_multiple_of = ensure_multiple_of

        def __call__(self, x):
            if torch.is_tensor(x):
                # CHW tensor
                h, w = x.shape[-2:]
            else:
                # PIL.Image.Image
                h, w = x.height, x.width
            if w < h:
                scale_factor = self.lower_bound / w
            else:
                scale_factor = self.lower_bound / h
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            new_h -= new_h % self.ensure_multiple_of
            new_w -= new_w % self.ensure_multiple_of
            if new_h < self.lower_bound:
                new_h = self.lower_bound
            if new_w < self.lower_bound:
                new_w = self.lower_bound
            x = TF.resize(x, (new_h, new_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
            return x

    class Identity():
        def __call__(self, x):
            return x

    if normalize:
        if normalize_mode == "imagenet":
            normalize_transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize_transform = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize_transform = Identity()

    transform = Compose([
        ModConstraintLowerBoundResize(lower_bound, ensure_multiple_of=ensure_multiple_of),
        ToTensor(),
        normalize_transform,
    ])
    return transform


def _test_run():
    import argparse
    import torch.nn.functional as F
    import numpy as np
    """
    pytyon hubconf.py -i ./input.jpg -o ./output.png --pil
    pytyon hubconf.py -i ./input.jpg -o ./output.png --pil --metric
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="output image file")
    parser.add_argument("--encoder", type=str, default="vitb",
                        choices=["vits", "vitb", "vitl", "v2_vits", "v2_vitb", "v2_vitl"],
                        help="encoder for relative depth model")
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--remote", action="store_true", help="use remote repo")
    parser.add_argument("--reload", action="store_true", help="reload remote repo")
    parser.add_argument("--pil", action="store_true", help="use PIL instead of OpenCV")
    parser.add_argument("--metric", action="store_true", help="use metric depth model")
    parser.add_argument("--metric-model-type", default="indoor",
                        choices=["indoor", "outdoor",
                                 "hypersim_s", "hypersim_b", "hypersim_l", "hypersim",
                                 "vkitti_s", "vkitti_b", "vkitti_l", "vkitti"],
                        help="model_type for metric depth")
    args = parser.parse_args()
    metric_v2 = args.metric and ("hypersim" in args.metric_model_type or "vkitti" in args.metric_model_type)

    if args.metric and not args.pil:
        raise NotImplementedError("--metric requires --pil option")

    if not args.metric:
        model_name = "DepthAnything"
        model_kwargs = dict(encoder=args.encoder)
        transforms_pil_kwargs = {}
    else:
        if metric_v2:
            model_name = "DepthAnythingMetricDepthV2"
            model_kwargs = dict(model_type=args.metric_model_type)
            transforms_pil_kwargs = {}
        else:
            model_name = "DepthAnythingMetricDepth"
            model_kwargs = dict(model_type=args.metric_model_type, remove_prep=True)
            transforms_pil_kwargs = dict(normalize_mode="center")

    if not args.remote:
        model = torch.hub.load(".", model_name, **model_kwargs,
                               source="local", trust_repo=True).cuda()
        if args.pil:
            transforms = torch.hub.load(".", "transforms_pil",
                                        **transforms_pil_kwargs, source="local", trust_repo=True)
        else:
            transforms = torch.hub.load(".", "transforms_cv2", source="local", trust_repo=True)
    else:
        force_reload = bool(args.reload)
        model = torch.hub.load("nagadomi/Depth-Anything_iw3", model_name, **model_kwargs,
                               force_reload=force_reload, trust_repo=True).cuda()
        if args.pil:
            transforms = torch.hub.load("nagadomi/Depth-Anything_iw3", "transforms_pil",
                                        **transforms_pil_kwargs, trust_repo=True)
        else:
            transforms = torch.hub.load("nagadomi/Depth-Anything_iw3", "transforms_cv2", trust_repo=True)

    if args.pil:
        import PIL
        image = PIL.Image.open(args.input).convert("RGB")
        h, w = image.height, image.width
        image = transforms(image).unsqueeze(0).cuda()
    else:
        import cv2
        image = cv2.imread(args.input, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        image = transforms(image).unsqueeze(0).cuda()

    if args.fp16:
        model = model.half()
        image = image.half()

    with torch.inference_mode():
        depth = model(image)
        if not args.metric:
            depth = depth.unsqueeze(0)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        else:
            depth = depth.unsqueeze(0) if metric_v2 else depth["metric_depth"]
            """
            if "hypersim" in args.metric_model_type:
                depth = 1. - depth / 20
            elif "vkitti" in args.metric_model_type:
                depth = 1. - depth / 80
            """
            depth = depth.neg()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            if True:  # False
                c = 0.45  # 0.6-0.1, 0.6: indoor, 0.3: outdoor
                c1 = 1.0 + c
                min_v = c / c1
                depth = ((c / (c1 - depth)) - min_v) / (1.0 - min_v)

        depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)
        depth = depth.squeeze(dim=[0, 1])
        depth = torch.clamp(depth, 0, 1)

    if args.pil:
        import torchvision.transforms.functional as TF
        depth = TF.to_pil_image(depth.cpu())
        # No color map
        depth.save(args.output)
    else:
        depth = (depth * 255).cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        cv2.imwrite(args.output, depth_color)


if __name__ == "__main__":
    _test_run()
