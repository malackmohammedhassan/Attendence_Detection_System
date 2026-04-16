import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { Dashboard } from '@/pages/Dashboard'
import { Training } from '@/pages/Training'
import { LiveCompare } from '@/pages/LiveCompare'
import { Benchmark } from '@/pages/Benchmark'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <Layout>
              <Dashboard />
            </Layout>
          }
        />
        <Route
          path="/training"
          element={
            <Layout>
              <Training />
            </Layout>
          }
        />
        <Route
          path="/live-compare"
          element={
            <Layout>
              <LiveCompare />
            </Layout>
          }
        />
        <Route
          path="/benchmark"
          element={
            <Layout>
              <Benchmark />
            </Layout>
          }
        />
        {/* Catch-all → dashboard */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
