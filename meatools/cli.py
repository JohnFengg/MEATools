import argparse
import sys

from .subcomands import mea_proccess

def _run_web(args):
    """Launch the local web front-end (meatools_web.server)."""
    import os as _os
    import meatools as _mt
    # meatools_web lives next to the meatools package (repo root); make it
    # importable regardless of how meatools itself was installed.
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(_mt.__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from meatools_web import server as _web
    argv = ["--port", str(args.port), "--host", args.host]
    if args.root:
        argv += ["--root", args.root]
    _web.main(argv)
    return 0

def main():
    parser=argparse.ArgumentParser(prog="mea", description="MEATOOLs CLI Toolkit")
    subparsers=parser.add_subparsers(dest="command")

    lte_parser=subparsers.add_parser("ttseq",help='Run test sequence analysis')
    lte_parser.set_defaults(func=mea_proccess.run_test_sequence)

    lte_parser=subparsers.add_parser("otr",help='Run OTR/Impedance analysis')
    lte_parser.set_defaults(func=mea_proccess.run_otr)

    lte_parser=subparsers.add_parser("ecsa",help='Run ECSA analysis')
    lte_parser.set_defaults(func=mea_proccess.run_ecsa)

    lte_parser=subparsers.add_parser("ecsadry",help='Run ECSA-dry analysis')
    lte_parser.set_defaults(func=mea_proccess.run_ecsa_dry)

    lte_parser=subparsers.add_parser("lsv",help='Run LSV analysis')
    lte_parser.set_defaults(func=mea_proccess.run_lsv)

    lte_parser=subparsers.add_parser("conclude",help='Extract key value from results directory')
    lte_parser.set_defaults(func=mea_proccess.run_conclude)

    lte_parser=subparsers.add_parser("all",help='Run all analysis')
    lte_parser.add_argument("--no-sulf",action="store_true",
                            help='Skip the sulf-cvrg step even if coverage '
                                 'data folders are present')
    lte_parser.set_defaults(func=mea_proccess.run_all)


    lte_parser=subparsers.add_parser("eis",help='Run EIS analysis')
    lte_parser.set_defaults(func=mea_proccess.run_eis)

    render_parser=subparsers.add_parser("render",help='Render results.json to HTML')
    render_parser.set_defaults(func=mea_proccess.run_render)

    sulf_cvrg_parser=subparsers.add_parser("sulf-cvrg",help='Calculate sulfonate group coverage')
    sulf_cvrg_parser.set_defaults(func=mea_proccess.run_sulfonate_coverage)

    web_parser=subparsers.add_parser("web",help='Run the local web front-end (upload + run + browse)')
    web_parser.add_argument("--port",type=int,default=8710,
                            help='listen port (default 8710)')
    web_parser.add_argument("--host",default="127.0.0.1",
                            help='listen address (default 127.0.0.1)')
    web_parser.add_argument("--root",default=None,
                            help='task root dir (default: '
                                 '/home/hrl/work/mea/mea_web or '
                                 '$MEATOOLS_WEB_ROOT)')
    web_parser.set_defaults(func=_run_web)


    # Everything after 'sulf-cvrg' (case dirs, --output, --port,
    # --non-interactive) is forwarded to the subcommand. parse_known_args
    # is used only for this command so the other subcommands stay strict
    # (argparse REMAINDER in subparsers mishandles leading options, B12).
    if len(sys.argv) > 1 and sys.argv[1] == 'sulf-cvrg':
        args, extra = parser.parse_known_args()
        args.sulf_args = list(extra)
    else:
        args = parser.parse_args()
        args.sulf_args = []
    if hasattr(args,'func'):
        rc = args.func(args) or 0
        sys.exit(rc)
    else:
        parser.print_help()