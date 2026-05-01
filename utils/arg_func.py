import argparse

# def basic_args():
#     parser = argparse.ArgumentParser(add_help=False)
#     parser.add_argument('--case_name', default = "AWA2_Baseline", type = str)
#     parser.add_argument("--model", type = str, required = True)
#     parser.add_argument("--basic_model", type = str, required = True)
#     parser.add_argument('--device', default = "0", type = str)
#     parser.add_argument('--cha', default = [32, 32, 32, 32], type = int, nargs='+')
#     parser.add_argument('--concept_per_layer', default = [32, 32, 32, 32], type = int, nargs='+')
#     parser.add_argument('--sel_layers', default = [2, 5, 8, 11], type = int, nargs='+')
#     parser.add_argument("--wo_cls", default = False, action = "store_true")
#     return parser

# def visualize_args():
#     parser = argparse.ArgumentParser(add_help=False)
#     parser.add_argument('--case_name', default = "AWA2_Baseline", type = str)
#     parser.add_argument("--model", type = str, required = True)
#     parser.add_argument("--basic_model", type = str, required = True)
#     parser.add_argument('--device', default = "0", type = str)
#     parser.add_argument('--cha', default = [32, 32, 32, 32], type = int, nargs='+')
#     parser.add_argument('--concept_per_layer', default = [32, 32, 32, 32], type = int, nargs='+')
#     return parser

# def dist_args():
#     parser = argparse.ArgumentParser(add_help=False)
#     parser.add_argument("--wo_norm", action = "store_true", default = False)
#     parser.add_argument("--MCP_norm", default = "normal", choices = ["normal", "softmax", "softmax_split"])
#     parser.add_argument("--MCP_temperature", default = 1.0, type = float)
#     return parser