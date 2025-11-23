export const stylesDarkTextDisplay = {
  container: {
    backgroundColor: '#1e1e1e',
    color: '#ffffff',
    minHeight: '100vh',
    padding: '20px',
    fontFamily: 'Arial, sans-serif'
  },
  header: {
    marginBottom: '20px'
  },
  inputRow: {
    display: 'flex',
    gap: '10px',
    alignItems: 'center'
  },
  textField: {
    flex: 1,
    padding: '10px',
    backgroundColor: '#2d2d2d',
    color: '#ffffff',
    border: '1px solid #444',
    borderRadius: '4px',
    fontSize: '16px'
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 'bold' as const,
    transition: 'all 0.2s ease-in-out',
    transform: 'translateY(0)',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)'
  },
  displayArea: {
    marginTop: '20px',
    padding: '15px',
    backgroundColor: '#2d2d2d',
    borderRadius: '4px',
    borderLeft: '4px solid #4CAF50',
    minHeight: '50px',
  }
};