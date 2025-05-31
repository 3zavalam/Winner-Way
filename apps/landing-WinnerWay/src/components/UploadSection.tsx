import React, { useState } from 'react';
import { Upload } from 'lucide-react';
import { useSession } from '@/context/SessionContext';
import { toast } from 'sonner';
import DemoSection from './DemoSection';
import ReferralCodeCard from './ReferralCodeCard';

const UploadSection: React.FC = () => {
  const { user } = useSession();

  const trialEnd = user?.trial_end ? new Date(user.trial_end) : null;
  const now = new Date();
  const isTrialActive = trialEnd && trialEnd > now;

  if (!isTrialActive) {
    toast.warning("Your free trial has ended. Scroll down to upgrade.");
    const section = document.getElementById("start-trial");
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
    }
    return null;
  }

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [strokeType, setStrokeType] = useState<'forehand' | 'backhand' | 'serve'>('forehand');
  const [handedness, setHandedness] = useState<'right' | 'left'>('right');
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [referenceUrl, setReferenceUrl] = useState<string | null>(null);
  const [keyframes, setKeyframes] = useState<{ [key: string]: string } | null>(null);
  const [analysis, setAnalysis] = useState<string[]>([]);
  const [drills, setDrills] = useState<
    { title: string; drill: string; steps: string[] }[]
  >([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!videoFile || !user?.email) return;

    setLoading(true);
    setFeedback(null);
    setVideoUrl(null);
    setKeyframes(null);
    setReferenceUrl(null);

    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('email', user.email);
    formData.append('stroke_type', strokeType);
    formData.append('handedness', handedness);

    try {
        const api = import.meta.env.VITE_BACKEND_URL;
        const res = await fetch(`${api}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setFeedback(data.error || 'Something went wrong processing your video.');
        setAnalysis([]);
        setDrills([]);
      } else {
        setFeedback(null);
        setVideoUrl(data.video_url || null);
        setKeyframes(data.keyframes || null);
        setReferenceUrl(data.reference_url || null);

        const feedbackLines = Array.isArray(data.feedback)
          ? data.feedback
          : (data.feedback || '')
              .split(/\n+/)
              .map(line => line.trim())
              .filter(Boolean);
        setAnalysis(feedbackLines);
        setDrills(Array.isArray(data.drills) ? data.drills : []);
      }
    } catch (err) {
      console.error(err);
      setFeedback('Error sending video.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="section" id="upload-section">
      <div className="winner-container max-w-5xl">
        <div className="bg-white/90 rounded-3xl p-8 md:p-12 shadow-xl border border-winner-green/20">
          <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4 text-center">
            Step 1: Upload Your Stroke
          </h2>

          <div className="mb-6">
            <p className="text-sm text-winner-green/80 mb-2 flex items-center gap-2">
              🎬 Not sure how to record? Watch this quick guide:
            </p>
            <video
              controls
              className="w-full max-w-md mx-auto rounded-lg shadow-md border border-winner-green/10"
            >
              <source src="/video-recording-guide.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>

          <div className="upload-area h-64 mb-8 relative cursor-pointer group">
            <input
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              aria-label="Upload video"
            />
            <div className="flex flex-col justify-center items-center h-full pointer-events-none">
              <Upload className="h-12 w-12 text-winner-green/60 mb-4 group-hover:text-winner-green" />
              <p className="text-center text-winner-green/80 text-lg font-medium mb-2">
                Drag & drop your tennis video here
              </p>
              <p className="text-center text-winner-green/60 text-sm">
                or click to upload
              </p>
              {videoFile && (
                <p className="text-center text-sm mt-2 text-winner-green font-semibold">
                  Selected: {videoFile.name}
                </p>
              )}
            </div>
          </div>

          <div className="mb-4 grid grid-cols-1 md:grid-cols-[75%_25%] gap-4">
            <div>
              <label className="block text-sm font-medium text-winner-green/80 mb-1">
                Which stroke do you want to analyze?
              </label>
              <select
                className="w-full border rounded p-2 text-winner-green"
                value={strokeType}
                onChange={e => setStrokeType(e.target.value as any)}
              >
                <option value="forehand">Forehand</option>
                <option value="backhand">Backhand</option>
                <option value="serve">Serve</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-winner-green/80 mb-1">
                Handedness
              </label>
              <select
                className="w-full border rounded p-2 text-winner-green"
                value={handedness}
                onChange={e => setHandedness(e.target.value as 'right' | 'left')}
              >
                <option value="right">Right-handed</option>
                <option value="left">Left-handed</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-4">
            <div>
              <p className="text-sm text-winner-green/70">
                Logged in as: <strong>{user?.email}</strong>
              </p>
            </div>
            <div className="self-end">
              <button
                onClick={handleAnalyze}
                disabled={loading || !videoFile}
                className="btn-primary w-full md:w-auto"
              >
                {loading ? 'Analyzing...' : 'Analyze Now'}
              </button>
            </div>
          </div>

          {feedback && (
            <div className="mt-4 text-center text-red-600 font-medium">
              {feedback}
            </div>
          )}

          {videoUrl && (
            <div className="mt-6 flex flex-col items-center gap-4">
              <video
                src={videoUrl}
                controls
                className="max-w-full rounded-xl shadow-md"
              />
              <div className="flex gap-4 items-center">
                <a href={videoUrl} download className="btn-secondary">
                  Download Analyzed Clip
                </a>
                <span className="text-sm text-winner-green/70">
                  Scroll down to see detailed results and drills!
                </span>
              </div>
            </div>
          )}
        </div>

        {videoUrl && (
          <>
            <DemoSection
              videoUrl={videoUrl}
              referenceUrl={referenceUrl || undefined}
              analysis={analysis}
              drills={drills}
            />
            <ReferralCodeCard userId={user.id} />
          </>
        )}
      </div>
    </section>
  );
};

export default UploadSection;