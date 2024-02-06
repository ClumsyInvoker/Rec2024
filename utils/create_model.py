from models.DSSM import DSSM_SASRec, DSSM_SASRec_PTCR

# baseline
from models.DIN import DeepInterestNetwork
from models.DeepFM import DeepFM
from models.DCN import DCN
from models.DropoutNet import DropoutNet
from models.SASRec import SASRec
from models.PPR import PPR
from models.CB2CF import CB2CF

from models.MetaEmb import MetaEmb

from models.PLATE import PLATE

# ablation
from models.DSSM_head import DSSM_SASRec_PTCR_head
from models.DSSM_net import DSSM_SASRec_PTCR_net


def check_config(model_name, config):
    assert "user_id" in config["dim_config"]
    assert "item_id" in config["dim_config"]

    if model_name == "DSSM_SASRec_PTCR":
        assert "prompt_embed_dim" in config and config["prompt_embed_dim"] is not None
        assert "prompt_net_hidden_size" in config and config["prompt_net_hidden_size"] is not None

    if model_name == "DSSM_SASRec_PTCR_head" or model_name == "DSSM_SASRec_PTCR_net":
        assert "maxlen" in config
        assert "prompt_embed_dim" in config and config["prompt_embed_dim"] is not None
        assert "prompt_net_hidden_size" in config and config["prompt_net_hidden_size"] is not None
        assert "prompt_ablation_setting" in config and config["prompt_ablation_setting"] is not None

    if model_name == "SASRec":
        assert "maxlen" in config
        assert "embed_dim" in config
        assert "dim_config" in config


def create_model(model_name, args, config):
    check_config(model_name, config)

    if model_name == "DSSM_SASRec":
        model = DSSM_SASRec(config).to(args.device)
    elif model_name == "DSSM_SASRec_PTCR":
        model = DSSM_SASRec_PTCR(config).to(args.device)
    elif model_name == "DeepFM":
        model = DeepFM(config).to(args.device)
    elif model_name == "DCN":
        model = DCN(config).to(args.device)
    elif model_name == "DropoutNet":
        model = DropoutNet(config).to(args.device)
    elif model_name == "SASRec":
        model = SASRec(config).to(args.device)
    elif model_name == "PPR":
        model = PPR(config).to(args.device)
    elif model_name == "CB2CF":
        model = CB2CF(config).to(args.device)
    elif model_name == "MetaEmb":
        model = MetaEmb(config).to(args.device)
    elif model_name == "PLATE":
        model = PLATE(config).to(args.device)
    elif model_name == "DSSM_SASRec_PTCR_head":
        model = DSSM_SASRec_PTCR_head(config).to(args.device)
    elif model_name == "DSSM_SASRec_PTCR_net":
        model = DSSM_SASRec_PTCR_net(config).to(args.device)
    else:
        raise Exception("No such model!")

    return model