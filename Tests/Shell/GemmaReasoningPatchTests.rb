require 'tmpdir'
require 'fileutils'
require 'digest'

root = File.expand_path('../..', __dir__)
checkout = File.join(root, '.build/checkouts/mlx-swift-lm')
patches = %w[mlx-swift-lm-reasoning-stream.patch mlx-swift-lm-gemma4-dense-fusion.patch mlx-swift-lm-gemma4-window-mask.patch mlx-swift-lm-gemma4-window-slicing.patch mlx-swift-lm-gemma4-text-mtp.patch mlx-swift-lm-gemma4-text-assistant.patch]
paths = %w[Libraries/MLXLMCommon/Evaluate.swift Libraries/MLXLMCommon/Tool/TokenStreamDecoder.swift Libraries/MLXLLM/Models/Gemma4Text.swift Libraries/MLXLLM/Models/Gemma4.swift Libraries/MLXLMCommon/MTPDrafterModel.swift Libraries/MLXVLM/Models/Gemma4Assistant.swift Libraries/MLXVLM/Models/Gemma4.swift Libraries/MLXLLM/Models/Gemma4TextAssistant.swift]

patches << 'mlx-swift-lm-gemma4-expert-gate-up.patch'

Dir.mktmpdir('gemma-patch-roundtrip') do |fixture|
  paths.each do |path|
    destination = File.join(fixture, path)
    FileUtils.mkdir_p(File.dirname(destination))
    FileUtils.cp(File.join(checkout, path), destination)
  end
  before = paths.map { |p| Digest::SHA256.file(File.join(fixture, p)).hexdigest }
  # Preparation reruns against the fully patched checkout. Later patches must
  # not invalidate the individual reverse checks used to detect earlier ones.
  patches.each do |patch|
    abort "Cannot recognize #{patch} in final stack" unless system('git', '-C', fixture,
      'apply', '--reverse', '--check', File.join(root, 'Patches', patch))
  end
  patches.reverse_each do |patch|
    abort "Cannot reverse #{patch}" unless system('git', '-C', fixture, 'apply', '--reverse', File.join(root, 'Patches', patch))
  end
  patches.each do |patch|
    path = File.join(root, 'Patches', patch)
    abort "Cannot apply #{patch}" unless system('git', '-C', fixture, 'apply', path)
    # The preparation script uses this check to recognize an already applied patch.
    abort "Cannot detect applied #{patch}" unless system('git', '-C', fixture, 'apply', '--reverse', '--check', path)
  end
  after = paths.map { |p| Digest::SHA256.file(File.join(fixture, p)).hexdigest }
  abort 'Patch roundtrip changed source bytes' unless before == after
end
puts 'Gemma dependency patch roundtrip passed'
