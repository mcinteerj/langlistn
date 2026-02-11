import AppKit
import Foundation
import ScreenCaptureKit
import AVFoundation

// MARK: - CLI Argument Parsing

struct Args {
    var appName: String?
    var sampleRate: Int = 16000
    var channels: Int = 1
    var listApps: Bool = false
}

func parseArgs() -> Args {
    var args = Args()
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        switch argv[i] {
        case "--app":
            i += 1; args.appName = argv[i]
        case "--rate":
            i += 1; args.sampleRate = Int(argv[i]) ?? 16000
        case "--channels":
            i += 1; args.channels = Int(argv[i]) ?? 1
        case "--list-apps":
            args.listApps = true
        default:
            break
        }
        i += 1
    }
    return args
}

// MARK: - Audio Capture Delegate

class AudioOutputHandler: NSObject, SCStreamOutput {
    let sampleRate: Int
    let channels: Int

    init(sampleRate: Int, channels: Int) {
        self.sampleRate = sampleRate
        self.channels = channels
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
            return
        }

        var length = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)

        guard let dataPointer = dataPointer, length > 0 else { return }

        guard let formatDesc = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(formatDesc) else { return }

        let srcSampleRate = asbd.pointee.mSampleRate
        let srcChannels = Int(asbd.pointee.mChannelsPerFrame)
        let bytesPerFrame = Int(asbd.pointee.mBytesPerFrame)
        let isFloat = asbd.pointee.mFormatFlags & kAudioFormatFlagIsFloat != 0
        let isNonInterleaved = asbd.pointee.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        let frameCount = length / bytesPerFrame

        // Convert to int16 mono
        var pcmData = Data()
        pcmData.reserveCapacity(frameCount * 2)

        if isFloat {
            if isNonInterleaved {
                // Non-interleaved: each channel is contiguous in the buffer
                // bytesPerFrame = 4 (one float per frame per channel)
                // Total data = frameCount * 4 * srcChannels (channels laid out sequentially)
                let totalFloats = length / 4
                let framesPerChannel = totalFloats / srcChannels
                let floatPtr = UnsafeRawPointer(dataPointer).bindMemory(to: Float.self, capacity: totalFloats)
                for f in 0..<framesPerChannel {
                    var sample: Float = 0
                    for c in 0..<srcChannels {
                        sample += floatPtr[c * framesPerChannel + f]
                    }
                    sample /= Float(srcChannels)
                    let clamped = max(-1.0, min(1.0, sample))
                    var int16Val = Int16(clamped * 32767.0)
                    pcmData.append(Data(bytes: &int16Val, count: 2))
                }
            } else if bytesPerFrame == srcChannels * 4 {
                // Interleaved float32
                let floatPtr = UnsafeRawPointer(dataPointer).bindMemory(to: Float.self, capacity: frameCount * srcChannels)
                for f in 0..<frameCount {
                    var sample: Float = 0
                    for c in 0..<srcChannels {
                        sample += floatPtr[f * srcChannels + c]
                    }
                    sample /= Float(srcChannels)
                    let clamped = max(-1.0, min(1.0, sample))
                    var int16Val = Int16(clamped * 32767.0)
                    pcmData.append(Data(bytes: &int16Val, count: 2))
                }
            }
        } else {
            pcmData.append(Data(bytes: dataPointer, count: length))
        }

        let actualFrameCount = pcmData.count / 2

        // Resample if needed
        if Int(srcSampleRate) != sampleRate && actualFrameCount > 0 {
            let ratio = srcSampleRate / Double(sampleRate)
            let outFrames = Int(Double(actualFrameCount) / ratio)
            var resampled = Data()
            resampled.reserveCapacity(outFrames * 2)
            for i in 0..<outFrames {
                let srcIdx = Int(Double(i) * ratio)
                let byteOffset = srcIdx * 2
                if byteOffset + 1 < pcmData.count {
                    resampled.append(pcmData[byteOffset])
                    resampled.append(pcmData[byteOffset + 1])
                }
            }
            pcmData = resampled
        }

        pcmData.withUnsafeBytes { ptr in
            let _ = fwrite(ptr.baseAddress, 1, ptr.count, stdout)
            fflush(stdout)
        }
    }
}

// MARK: - Stream Delegate

class StreamDelegate: NSObject, SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("Stream stopped with error: \(error.localizedDescription)\n", stderr)
    }
}

// MARK: - App Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    let args: Args
    var stream: SCStream?
    var handler: AudioOutputHandler?
    var streamDelegate: StreamDelegate?

    init(args: Args) {
        self.args = args
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide from Dock
        NSApp.setActivationPolicy(.accessory)

        Task {
            await self.run()
        }
    }

    func run() async {
        do {
            let content = try await SCShareableContent.current

            if args.listApps {
                let apps = content.applications
                    .filter { $0.applicationName.count > 0 }
                    .map { $0.applicationName }
                let unique = Array(Set(apps)).sorted()
                for name in unique {
                    print(name)
                }
                exit(0)
            }

            guard let appName = args.appName else {
                fputs("Error: --app required\n", stderr)
                exit(1)
            }

            guard let app = content.applications.first(where: {
                $0.applicationName.localizedCaseInsensitiveContains(appName)
            }) else {
                fputs("Error: App '\(appName)' not found or not running\n", stderr)
                exit(1)
            }

            let appFilter = SCContentFilter(
                display: content.displays[0],
                including: [app],
                exceptingWindows: []
            )

            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.excludesCurrentProcessAudio = true
            config.sampleRate = 48000
            config.channelCount = 2
            config.width = 2
            config.height = 2
            config.minimumFrameInterval = CMTime(value: 1, timescale: 1)  // 1 FPS — we only need audio

            let streamDel = StreamDelegate()
            self.streamDelegate = streamDel
            let s = SCStream(filter: appFilter, configuration: config, delegate: streamDel)
            let h = AudioOutputHandler(sampleRate: args.sampleRate, channels: args.channels)
            self.handler = h
            self.stream = s

            // Must add .screen output too — audio callbacks won't fire without it
            try s.addStreamOutput(h, type: .screen, sampleHandlerQueue: .global())
            try s.addStreamOutput(h, type: .audio, sampleHandlerQueue: .global())

            try await s.startCapture()
            fputs("Capturing audio from: \(app.applicationName)\n", stderr)

        } catch {
            let msg = error.localizedDescription
            if msg.contains("permission") || msg.contains("denied") || msg.contains("TCCDeny") {
                fputs("Error: Screen Recording permission denied. Grant in System Settings > Privacy.\n", stderr)
                exit(3)
            }
            fputs("Error: \(msg)\n", stderr)
            exit(1)
        }
    }
}

// MARK: - Main

let args = parseArgs()

signal(SIGTERM) { _ in exit(0) }
signal(SIGINT) { _ in exit(0) }

let delegate = AppDelegate(args: args)
let app = NSApplication.shared
app.delegate = delegate
app.run()
