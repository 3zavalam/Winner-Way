import React from 'react';
import Header from '../components/Header';
import Hero from '../components/Hero';
import UploadSection from '../components/UploadSection';
import HowItWorks from '../components/HowItWorks';
import Footer from '../components/Footer';
import LoginFormWithBuddyPass from '../components/LoginFormWithBuddyPass';
import StartTrialPromo from '../components/StartTrialPromo';
import { useSession } from '../context/SessionContext';

const Index = () => {
  const { user, trialActive, loading } = useSession();

  return (
    <div className="min-h-screen flex flex-col bg-winner-beige">
      <Header />
      <main>
        <Hero />

        {/* Mostrar loading temporal */}
        {loading ? (
          <div className="text-center py-12 text-winner-green">Loading...</div>
        ) : !user ? (
          <LoginFormWithBuddyPass />
        ) : !trialActive ? (
          <StartTrialPromo />
        ) : (
          <UploadSection />
        )}

        <HowItWorks />
      </main>
      <Footer />
    </div>
  );
};

export default Index;

