#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
fetch() {
  local name=$1 url=$2 commit=$3 path=".upstream/$1"
  if [[ ! -d "$path/.git" ]]; then git clone --filter=blob:none --no-checkout "$url" "$path"; fi
  git -C "$path" fetch --depth 1 origin "$commit"
  git -C "$path" checkout --detach "$commit"
  [[ $(git -C "$path" rev-parse HEAD) == "$commit" ]]
  echo "$name locked at $commit"
}
fetch MambaVision https://github.com/NVlabs/MambaVision 7860a506b2eb844eaaae676f08461ce8c3c26f43
fetch TransNeXt https://github.com/DaiShiResearch/TransNeXt c8a99743b60ac94ac8d2bf66ffe164a440dcfe21
fetch CheXWorld https://github.com/LeapLabTHU/CheXWorld 090102758801dc097f53c49d135b835570c8d173
fetch LRFL https://github.com/Statistical-Deep-Learning/LRFL 6c5ce47fd99cbaec915dde8db71d83d03c64423d
fetch X-WIN https://github.com/RPIDIAL/X-WIN 6cc45859a07674c4efeb81e7b1d06f8187d206c8
fetch CARZero https://github.com/laihaoran/CARZero fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d
