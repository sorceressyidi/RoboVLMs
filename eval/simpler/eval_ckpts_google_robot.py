import os

ckpt_paths = [
    (
        "path/to/VLA-Checkpoint-{epoch}-{steps}.ckpt",
        "path/to/VLA-Checkpoint-config.json",
    )
]

for i, (ckpt, config) in enumerate(ckpt_paths):
    print("evaluating checkpoint {}".format(ckpt))
    os.system(
        "bash bash/simpler_eval/openvla_pick_coke_can_visual_matching.sh {} {}".format(
            ckpt, config
        )
    )
    os.system(
        "bash bash/simpler_eval/openvla_move_near_visual_matching.sh {} {}".format(
            ckpt, config
        )
    )
    os.system(
        "bash bash/simpler_eval/openvla_put_in_drawer_visual_matching.sh {} {}".format(
            ckpt, config
        )
    )
    os.system(
        "bash bash/simpler_eval/openvla_drawer_visual_matching.sh {} {}".format(
            ckpt, config
        )
    )
