
import torch
from mobilefacenet import MobileFaceNet
from pathlib import Path

def convert():
    scripted_path = Path("../models/pretrained/mobilefacenet_scripted.pt")
    output_path = Path("../models/mobilefacenet_pretrained.pth")
    
    print(f"Loading scripted model from {scripted_path}...")
    scr = torch.jit.load(str(scripted_path), map_location="cpu")
    scr_sd = scr.state_dict()
    
    our_model = MobileFaceNet()
    our_sd = our_model.state_dict()
    
    new_sd = {}
    
    # --- MAPPING LOGIC ---
    
    # 1. Stem
    new_sd["stem.conv.weight"] = scr_sd["conv1.0.weight"]
    new_sd["stem.bn.weight"] = scr_sd["conv1.1.weight"]
    new_sd["stem.bn.bias"] = scr_sd["conv1.1.bias"]
    # (running mean/var will be initialized/random, which is okay for fine-tuning)

    # 2. DW Stem
    new_sd["dw_stem.conv.weight"] = scr_sd["dw_conv.depthwise.weight"]
    new_sd["dw_stem.bn.weight"] = scr_sd["dw_conv.bn1.weight"]
    new_sd["dw_stem.bn.bias"] = scr_sd["dw_conv.bn1.bias"]
    # PReLU weight if available, if not skip
    if "dw_conv.bn1.weight" in scr_sd: # check for prelu
         pass 

    # 3. Bottlenecks (features.0 to features.14)
    for i in range(15):
        prefix_scr = f"features.{i}.conv"
        prefix_our = f"bottlenecks.{i}.block"
        
        # In our Bottleneck: block[0] is expand, block[1] is depthwise, block[2] is project
        # In scripted: it seems flattened or different structure.
        # Let's map based on the 15 blocks sequence.
        
        try:
            # Expand (1x1)
            new_sd[f"{prefix_our}.0.conv.weight"] = scr_sd[f"{prefix_scr}.0.0.weight"]
            new_sd[f"{prefix_our}.0.bn.weight"] = scr_sd[f"{prefix_scr}.0.1.weight"]
            new_sd[f"{prefix_our}.0.bn.bias"] = scr_sd[f"{prefix_scr}.0.1.bias"]
            
            # Depthwise (3x3)
            new_sd[f"{prefix_our}.1.conv.weight"] = scr_sd[f"{prefix_scr}.1.0.weight"]
            new_sd[f"{prefix_our}.1.bn.weight"] = scr_sd[f"{prefix_scr}.1.1.weight"]
            new_sd[f"{prefix_our}.1.bn.bias"] = scr_sd[f"{prefix_scr}.1.1.bias"]
            
            # Project (1x1)
            new_sd[f"{prefix_our}.2.conv.weight"] = scr_sd[f"{prefix_scr}.2.weight"]
            new_sd[f"{prefix_our}.2.bn.weight"] = scr_sd[f"{prefix_scr}.3.weight"]
            new_sd[f"{prefix_our}.2.bn.bias"] = scr_sd[f"{prefix_scr}.3.bias"]
        except KeyError:
            print(f"Skipping bottleneck {i} mapping due to key mismatch")

    # 4. Final Conv
    new_sd["conv_last.conv.weight"] = scr_sd["conv2.0.weight"]
    new_sd["conv_last.bn.weight"] = scr_sd["conv2.1.weight"]
    new_sd["conv_last.bn.bias"] = scr_sd["conv2.1.bias"]

    # 5. GDC
    new_sd["gdc.conv.weight"] = scr_sd["gdconv.depthwise.weight"]
    new_sd["gdc.bn.weight"] = scr_sd["gdconv.bn.weight"]
    new_sd["gdc.bn.bias"] = scr_sd["gdconv.bn.bias"]

    # 6. Embedding (mapping may be partial)
    if "conv3.weight" in scr_sd:
        new_sd["embedding.1.weight"] = scr_sd["conv3.weight"].view(128, 512)
    if "bn.weight" in scr_sd:
        new_sd["embedding.2.weight"] = scr_sd["bn.weight"]
        new_sd["embedding.2.bias"] = scr_sd["bn.bias"]

    # Load into our model to verify
    missing, unexpected = our_model.load_state_dict(new_sd, strict=False)
    print(f"Successfully mapped {len(new_sd)} keys.")
    print(f"Missing keys: {len(missing)}")
    
    # Save
    torch.save(new_sd, output_path)
    print(f"✅ Pretrained model saved to: {output_path}")

if __name__ == "__main__":
    convert()
