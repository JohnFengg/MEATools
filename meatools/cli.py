import argparse
import sys

from .subcomands import mea_proccess

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