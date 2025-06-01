import React from 'react';
import Header from '../components/Header';
import Hero from '../components/Hero';
import UploadSection from '../components/UploadSection';
import HowItWorks from '../components/HowItWorks';
import Footer from '../components/Footer';
import StartTrialPromo from '../components/StartTrialPromo';
import BuyNowSection from '../components/BuyNowSection';
import { useSession } from '../context/SessionContext';

const Index = () => {
  const { user, trialActive, loading } = useSession();

  return (
    <div className="min-h-screen flex flex-col bg-winner-beige">
      <Header />
      <main>
        <Hero />
        <UploadSection />
        <BuyNowSection />
        <HowItWorks />
        {/* Only show trial promo if user is logged in but trial is not active */}
        {!loading && user && !trialActive && <StartTrialPromo />}
      </main>
      <Footer />
    </div>
  );
};

export default Index;

