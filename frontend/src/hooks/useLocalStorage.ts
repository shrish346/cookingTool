import { useState, useEffect } from 'react'

/**
 * Hook for persisting state to localStorage
 */
export function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T | ((prev: T) => T)) => void] {
  // Get initial value from localStorage or use default
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? JSON.parse(item) : initialValue
    } catch (error) {
      console.warn(`Error reading localStorage key "${key}":`, error)
      return initialValue
    }
  })

  // Update localStorage when state changes
  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(storedValue))
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error)
    }
  }, [key, storedValue])

  return [storedValue, setStoredValue]
}

/**
 * Hook specifically for storing the last recipe
 */
export function useSavedRecipe() {
  return useLocalStorage<{
    recipe: import('../types').Recipe | null
    savedAt: string | null
  }>('chefs-loop-recipe', { recipe: null, savedAt: null })
}


