import React from 'react';

interface DrillItem {
  title: string;
  drill: string;
  steps: string[];
}

interface DemoSectionProps {
  videoUrl: string;
  referenceUrl?: string;
  analysis?: string[];
  drills?: DrillItem[];
}

const formatMarkdownToHtml = (text: string): string => {
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
};

const DemoSection: React.FC<DemoSectionProps> = ({
  videoUrl,
  referenceUrl,
  analysis = [],
  drills = []
}) => {
  return (
    <section className="section bg-white/50" id="demo">
      <div className="winner-container max-w-6xl">
        <h2 className="text-3xl font-bold text-winner-green mb-3 text-center">
          Step 2: See the Results
        </h2>
        <p className="text-xl text-winner-green/70 mb-12 text-center max-w-3xl mx-auto">
          Your stroke. Compared to the pros. Here's how you'll improve.
        </p>

        <div className="bg-white rounded-2xl p-6 md:p-8 shadow-lg">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="aspect-video bg-winner-green/10 rounded-xl mb-4 overflow-hidden">
                <video
                  src={videoUrl}
                  controls
                  className="w-full h-full object-cover"
                />
              </div>
              <h3 className="font-semibold text-lg text-winner-green">
                Your Technique
              </h3>
              <p className="text-winner-green/70">
                Key points identified from your stroke
              </p>
            </div>

            <div>
              <div className="aspect-video bg-winner-green/10 rounded-xl mb-4 overflow-hidden">
                {referenceUrl ? (
                  <video
                    src={referenceUrl}
                    controls
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gray-100 text-winner-green/50 text-center px-4">
                    <p>
                      No reference video available for this stroke.
                      <br />
                      Please try a different one or check your settings.
                    </p>
                  </div>
                )}
              </div>
              <h3 className="font-semibold text-lg text-winner-green">
                Pro Reference
              </h3>
              <p className="text-winner-green/70">
                Perfect technique for comparison
              </p>
            </div>
          </div>

          {analysis.length > 0 && (
            <div className="mt-8 bg-winner-green/5 p-6 rounded-xl">
              <h3 className="font-bold text-xl text-winner-green mb-3">
                AI Analysis
              </h3>
              <div className="space-y-4">
                {analysis.map((point, idx) => {
                  const isPositive = point.startsWith('✔️');
                  const isNegative = point.startsWith('⚠️');
                  const bgColor = isPositive
                    ? 'bg-green-500'
                    : isNegative
                    ? 'bg-yellow-500'
                    : 'bg-winner-green';

                  return (
                    <div key={idx} className="flex items-start gap-3">
                      <span
                        className={`${bgColor} text-white font-bold rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5`}
                      >
                        {idx + 1}
                      </span>
                      <div className="text-winner-green/80 whitespace-pre-line">
                        {point}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {drills.length > 0 && (
            <div className="mt-8 bg-winner-green/5 p-6 rounded-xl">
              <h3 className="font-bold text-xl text-winner-green mb-3">
                Drills: How to Improve
              </h3>

              <div className="space-y-4">
                {drills.map((drill, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <span className="bg-winner-green text-white font-bold rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 mt-0.5">
                      {idx + 1}
                    </span>
                    <div className="text-winner-green/80">
                      <div className="font-semibold">{drill.title}</div>
                      <div className="italic mt-1">{drill.drill}</div>
                      <ul className="list-disc list-inside mt-2">
                        {drill.steps.map((step, stepIdx) => (
                          <li key={stepIdx}>{step}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

export default DemoSection;
