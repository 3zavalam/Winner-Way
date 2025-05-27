
import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';

const HowItWorks: React.FC = () => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <section className="section" id="how-it-works">
      <div className="winner-container max-w-4xl">
        <div className="text-center mb-8">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-2 mx-auto btn-secondary"
          >
            <span>How It Works</span>
            <ChevronDown 
              className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`} 
            />
          </button>
        </div>
        
        {isExpanded && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8 animate-fade-in">
            <div className="bg-white/80 p-6 rounded-2xl shadow-md flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-winner-green/10 rounded-full flex items-center justify-center mb-4">
                <span className="font-bold text-winner-green text-2xl">1</span>
              </div>
              <h3 className="font-bold text-xl text-winner-green mb-2">Upload Your Video</h3>
              <p className="text-winner-green/70">
                Upload your forehand, backhand or serve for analysis
              </p>
            </div>
            
            <div className="bg-white/80 p-6 rounded-2xl shadow-md flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-winner-green/10 rounded-full flex items-center justify-center mb-4">
                <span className="font-bold text-winner-green text-2xl">2</span>
              </div>
              <h3 className="font-bold text-xl text-winner-green mb-2">AI Analysis</h3>
              <p className="text-winner-green/70">
                We extract key points and compare to a pro's technique
              </p>
            </div>
            
            <div className="bg-white/80 p-6 rounded-2xl shadow-md flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-winner-green/10 rounded-full flex items-center justify-center mb-4">
                <span className="font-bold text-winner-green text-2xl">3</span>
              </div>
              <h3 className="font-bold text-xl text-winner-green mb-2">Get Feedback</h3>
              <p className="text-winner-green/70">
                You receive simple visual + text feedback to improve
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default HowItWorks;
