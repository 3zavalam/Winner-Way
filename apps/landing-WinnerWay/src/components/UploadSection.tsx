import React, { useState, useEffect } from "react";
import { Upload } from "lucide-react";
import { useSession } from "@/context/SessionContext";
import { toast } from "sonner";
import DemoSection from "./DemoSection";
import { supabase } from "@/lib/supabaseClient";

const UploadSection: React.FC = () => {
  const { user } = useSession();
  const [userEmail, setUserEmail] = useState<string | null>(() => {
    // Intentar recuperar el email del localStorage al cargar
    return localStorage.getItem('winner_way_email') || null;
  });
  const [showEmailForm, setShowEmailForm] = useState(false);
  const [emailInput, setEmailInput] = useState("");
  const [emailError, setEmailError] = useState("");
  const [remainingAnalyses, setRemainingAnalyses] = useState<number | null>(null);
  const [loadingAnalyses, setLoadingAnalyses] = useState(true);

  // ─── Estados internos para la subida y análisis del video ────────────────
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [strokeType, setStrokeType] = useState<"forehand" | "backhand" | "serve">("forehand");
  const [handedness, setHandedness] = useState<"right" | "left">("right");
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [referenceUrl, setReferenceUrl] = useState<string | null>(null);
  const [keyframes, setKeyframes] = useState<{ [key: string]: string } | null>(null);
  const [analysis, setAnalysis] = useState<string[]>([]);
  const [drills, setDrills] = useState<{ title: string; drill: string; steps: string[] }[]>([]);

  const STRIPE_LINK = "https://buy.stripe.com/test_9B65kF9aI08r0tY2SmgIo00";

  // ─── Verificar análisis restantes ────────────────────────────────────────
  const checkRemainingAnalyses = async (email: string): Promise<number | null> => {
    setLoadingAnalyses(true);
    let calculatedRemaining: number | null = null;
    try {
      const normalizedEmailForRPC = email.trim().toLowerCase();
      // console.log('[WW_DEBUG] Calling RPC check_daily_analysis_limit for email:', normalizedEmailForRPC);

      const { data: analysesToday, error: rpcError } = await supabase
        .rpc('check_daily_analysis_limit', { user_email: normalizedEmailForRPC });

      // console.log('[WW_DEBUG] RPC returned analysesToday:', analysesToday);
      // console.log('[WW_DEBUG] RPC error:', rpcError);

      if (rpcError) {
        // console.error('[WW_DEBUG] RPC call failed:', rpcError);
        throw rpcError;
      }

      const dailyLimit = 3;
      const count = (analysesToday === null || typeof analysesToday === 'undefined') ? 0 : Number(analysesToday);
      
      calculatedRemaining = dailyLimit - count;
      
      // console.log('[WW_DEBUG] Count from RPC:', count);
      // console.log('[WW_DEBUG] Calculated remaining analyses (RPC path):', calculatedRemaining);
      setRemainingAnalyses(calculatedRemaining);

    } catch (err) {
      // console.error('[WW_DEBUG] Error in checkRemainingAnalyses (using RPC):', err);
      toast.error('Error checking analysis limit. Please try again.');
      setRemainingAnalyses(null);
      calculatedRemaining = null;
    } finally {
      setLoadingAnalyses(false);
      return calculatedRemaining;
    }
  };

  // ─── Actualizar análisis restantes cuando cambia el email ────────────────
  useEffect(() => {
    const emailFromState = userEmail; // This is from localStorage initially, or from setEmailInput
    const emailFromSession = user?.email;
    let emailToUse: string | null = null;

    if (emailFromState) {
      const normalizedEmail = emailFromState.trim().toLowerCase();
      if (emailFromState !== normalizedEmail) {
        // emailFromState was not normalized. Normalize it in state and localStorage.
        // This will trigger a re-run of this useEffect.
        setUserEmail(normalizedEmail);
        localStorage.setItem('winner_way_email', normalizedEmail);
        return; // Exit early, wait for re-run with normalized userEmail
      }
      emailToUse = normalizedEmail;
    } else if (emailFromSession) {
      // If no userEmail from state/localStorage, use session email.
      // For safety, normalize it.
      emailToUse = emailFromSession.trim().toLowerCase();
    }

    if (emailToUse) {
      checkRemainingAnalyses(emailToUse);
    } else {
      setRemainingAnalyses(null); // No email available
      setLoadingAnalyses(false);
    }
  }, [userEmail, user?.email]);

  // ─── Validar email ───────────────────────────────────────────────────────
  const validateEmail = (email: string) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  };

  // ─── Manejar envío de email ─────────────────────────────────────────────
  const handleEmailSubmit = async () => {
    if (!validateEmail(emailInput)) {
      setEmailError("Please enter a valid email address");
      return;
    }
    const normalizedEmail = emailInput.trim().toLowerCase();

    setEmailError("");
    setUserEmail(normalizedEmail);
    setShowEmailForm(false);
    
    localStorage.setItem('winner_way_email', normalizedEmail);
    
    const currentRemaining = await checkRemainingAnalyses(normalizedEmail);
    
    if (currentRemaining !== null && currentRemaining <= 0) {
      // Redirigir a Stripe
      window.location.href = STRIPE_LINK;
    }
  };

  // ─── Manejar análisis ───────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!videoFile) {
      toast.error("Please upload a video first");
      return;
    }

    let emailForAnalysis: string | null = null;
    if (userEmail) {
        emailForAnalysis = userEmail;
    } else if (user?.email) {
        emailForAnalysis = user?.email.trim().toLowerCase();
    }

    if (!emailForAnalysis) {
      setShowEmailForm(true);
      return;
    }

    if (remainingAnalyses !== null && remainingAnalyses <= 0) {
      toast.error("You've reached your daily limit. Get unlimited access to continue!");
      // Redirigir a Stripe
      window.location.href = STRIPE_LINK;
      return;
    }

    setLoadingUpload(true);
    setFeedback(null);
    setVideoUrl(null);
    setKeyframes(null);
    setReferenceUrl(null);

    const emailToUse = emailForAnalysis;

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("email", emailToUse);
    formData.append("stroke_type", strokeType);
    formData.append("handedness", handedness);

    try {
      const api = import.meta.env.VITE_BACKEND_URL;
      const res = await fetch(`${api}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        setFeedback(data.error || "Something went wrong processing your video.");
        setAnalysis([]);
        setDrills([]);
      } else {
        const { error: logError } = await supabase
          .from('analysis_logs')
          .insert({
            email: emailToUse,
            stroke_type: strokeType,
            handedness: handedness,
            video_url: data.video_url,
            reference_url: data.reference_url,
            analysis: data.feedback,
            drills: data.drills
          });

        if (logError) {
          console.error('Error logging analysis:', logError);
        }

        const currentRemainingAfterAnalysis = await checkRemainingAnalyses(emailToUse); 

        setFeedback(null);
        setVideoUrl(data.video_url ? data.video_url.replace(/^http:/, 'https:') : null);
        setKeyframes(data.keyframes || null);
        setReferenceUrl(data.reference_url ? data.reference_url.replace(/^http:/, 'https:') : null);

        const feedbackLines = Array.isArray(data.feedback)
          ? data.feedback
          : (data.feedback || "")
              .split(/\n+/)
              .map((line: string) => line.trim())
              .filter(Boolean);
        setAnalysis(feedbackLines);
        setDrills(Array.isArray(data.drills) ? data.drills : []);
      }
    } catch (err) {
      console.error(err);
      setFeedback("Error sending video.");
    } finally {
      setLoadingUpload(false);
    }
  };

  // ─── 4. Handlers ────────────────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  return (
    <section className="section" id="upload-section">
      <div className="winner-container max-w-5xl">
        <div className="bg-white/90 rounded-3xl p-8 md:p-12 shadow-xl border border-winner-green/20">
          <h2 className="text-2xl md:text-3xl font-bold text-winner-green mb-4 text-center">
            Step 1: Upload Your Stroke
          </h2>

          {/* Mostrar análisis restantes si hay email */}
          {(userEmail || user?.email) && !loadingAnalyses && remainingAnalyses !== null && (
            <div className="text-center mb-6">
              <div className="text-winner-green/80">
                {remainingAnalyses <= 0 ? (
                  <div className="space-y-4">
                    <span className="text-red-500 font-medium block">
                      You've reached your daily limit of 3 analyses. Try again tomorrow!
                    </span>
                    <button
                      onClick={() => window.location.href = STRIPE_LINK}
                      className="btn-primary inline-block"
                    >
                      Get Unlimited Access
                    </button>
                  </div>
                ) : remainingAnalyses === 1 ? (
                  <div className="space-y-4">
                    <span className="block">
                      You have <strong>1</strong> analysis remaining today
                    </span>
                    <button
                      onClick={() => window.location.href = STRIPE_LINK}
                      className="btn-secondary inline-block"
                    >
                      Get Unlimited Access
                    </button>
                  </div>
                ) : (
                  <span>
                    You have <strong>{remainingAnalyses}</strong> analyses remaining today
                  </span>
                )}
              </div>
            </div>
          )}

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

          {/* Formulario de email inline - solo se muestra si no hay email guardado */}
          {showEmailForm && !userEmail && (
            <div className="mb-6 p-4 bg-winner-green/5 rounded-lg">
              <h3 className="text-lg font-medium text-winner-green mb-2">
                One last step before analysis
              </h3>
              <p className="text-sm text-winner-green/70 mb-3">
                Enter your email to receive your analysis results
              </p>
              <div className="flex gap-2">
                <input
                  type="email"
                  value={emailInput}
                  onChange={(e) => {
                    setEmailInput(e.target.value);
                    setEmailError("");
                  }}
                  placeholder="your@email.com"
                  className="flex-1 p-2 border rounded text-sm"
                />
                <button
                  onClick={handleEmailSubmit}
                  className="btn-primary whitespace-nowrap"
                >
                  Continue
                </button>
              </div>
              {emailError && (
                <p className="text-red-500 text-sm mt-1">{emailError}</p>
              )}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-4">
            <div>
              <p className="text-sm text-winner-green/70">
                {userEmail || user?.email ? (
                  <>Using email: <strong>{userEmail || user?.email}</strong></>
                ) : (
                  "Enter your email to start analyzing"
                )}
              </p>
            </div>
            <div className="self-end">
              <button
                onClick={handleAnalyze}
                disabled={loadingUpload || !videoFile || (remainingAnalyses !== null && remainingAnalyses <= 0)}
                className="btn-primary w-full md:w-auto"
              >
                {loadingUpload ? 'Analyzing...' : 'Analyze Now'}
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
          <DemoSection
            videoUrl={videoUrl}
            referenceUrl={referenceUrl || undefined}
            analysis={analysis}
            drills={drills}
          />
        )}
      </div>
    </section>
  );
};

export default UploadSection;