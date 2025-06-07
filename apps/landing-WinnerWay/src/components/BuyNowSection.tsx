import React from 'react';
import { Check } from 'lucide-react';

const BuyNowSection: React.FC = () => {
  const STRIPE_LINK = "https://buy.stripe.com/cNi14og9na0oeEDdZwbMQ01";

  const handleBuyNow = () => {
    window.location.href = STRIPE_LINK;
  };

  return (
    <section id="buy-now-section" className="section bg-winner-green/5">
      <div className="winner-container max-w-5xl">
        <div className="bg-white rounded-3xl p-8 md:p-12 shadow-xl border border-winner-green/20">
          <div className="text-center mb-8">
            <h2 className="text-3xl md:text-4xl font-bold text-winner-green mb-4">
              Get Unlimited Access
            </h2>
            <p className="text-lg text-winner-green/80 max-w-2xl mx-auto">
              Unlock unlimited video analyses and take your tennis game to the next level
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div className="space-y-6">
              <div className="flex items-start gap-3">
                <Check className="h-6 w-6 text-winner-green flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-winner-green">Unlimited Analyses</h3>
                  <p className="text-winner-green/70">No more daily limits. Analyze as many videos as you want.</p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <Check className="h-6 w-6 text-winner-green flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-winner-green">Priority Processing</h3>
                  <p className="text-winner-green/70">Get your analyses faster with priority queue access.</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Check className="h-6 w-6 text-winner-green flex-shrink-0 mt-1" />
                <div>
                  <h3 className="font-semibold text-winner-green">Detailed Feedback</h3>
                  <p className="text-winner-green/70">Receive comprehensive analysis and personalized drills.</p>
                </div>
              </div>
            </div>

            <div className="bg-winner-green/5 rounded-2xl p-8 text-center">
              <div className="mb-6">
                <span className="text-4xl font-bold text-winner-green">$29</span>
                <span className="text-winner-green/70">/year</span>
              </div>
              <p className="text-winner-green/70 mb-6">
                That's less than $2.99 per month to improve your tennis game
              </p>
              <button
                onClick={handleBuyNow}
                className="btn-primary w-full text-lg py-4"
              >
                Get Unlimited Access
              </button>
              <p className="text-sm text-winner-green/60 mt-4">
                30-day money-back guarantee
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default BuyNowSection; 