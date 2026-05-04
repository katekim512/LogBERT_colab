ABLATION_CHOICES = ("main", "semparser", "freq", "semantic_id")


def add_ablation_argument(parser):
    parser.add_argument(
        "--ablation",
        choices=ABLATION_CHOICES,
        default="main",
        help="Select ablation logic to run.",
    )


def is_semparser_like(ablation):
    return ablation in ("semparser", "freq", "semantic_id")


def is_freq_like(ablation):
    return ablation in ("freq", "semantic_id")


def is_semantic_id_like(ablation):
    return ablation == "semantic_id"


def apply_logbert_ablation(options, ablation):
    updated = dict(options)
    updated["ablation"] = ablation
    updated["use_sbert_embedding"] = is_semparser_like(ablation)
    updated["is_freq"] = is_freq_like(ablation)
    updated["use_semantic_id"] = is_semantic_id_like(ablation)
    if is_semantic_id_like(ablation):
        updated.setdefault("semantic_id_weight", 0.01)
    return updated
