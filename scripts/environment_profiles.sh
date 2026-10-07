#!/usr/bin/env bash
# Shared dependency groups used by installation and permanent registration.
nav_eval_environment_profile() {
    case "${1#nav_}" in
        streamvln|gavln)
            environment_profile=streamvln; python_version=3.9; requirements=gavln-inference.txt ;;
        vlm|vila|navid|uni-navid|uni_navid|navila|awarevln|internnav|janusvln|internvla-n1|internvla_n1|qwen_vl|qwen-vl|navida|activevln|onevla)
            environment_profile=vlm; python_version=3.10; requirements=vlm-inference.txt ;;
        habitat024|habitat030)
            environment_profile=${1#nav_}; python_version=3.9; requirements= ;;
        isaac|isaacsim500)
            environment_profile=isaac; python_version=3.11; requirements=isaac-inference.txt ;;
        *) echo "Unsupported environment profile: $1" >&2; return 2 ;;
    esac
    environment_name=nav_$environment_profile
}
