/**
 * API client for Chef's Loop backend
 */

import type { ValidateResponse, ProcessResponse, StatusResponse } from '../types'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}/api${endpoint}`
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Request failed')
  }
  
  return response.json()
}

export const api = {
  /**
   * Deep validate a YouTube or TikTok URL
   */
  async validateUrl(url: string): Promise<ValidateResponse> {
    return request<ValidateResponse>('/validate', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  },
  
  /**
   * Start processing a video URL
   */
  async processVideo(url: string): Promise<ProcessResponse> {
    return request<ProcessResponse>('/process', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  },
  
  /**
   * Get the status of a processing job
   */
  async getStatus(jobId: string): Promise<StatusResponse> {
    return request<StatusResponse>(`/status/${jobId}`)
  },
}

export { ApiError }

