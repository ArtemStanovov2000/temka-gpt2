import { createSlice } from "@reduxjs/toolkit";

type State = {
  context: string;
}

const initialState: State = {
  context: "",
}

const contextSlice = createSlice({
  name: "context",
  initialState,
  reducers: {
    addContext(state, action) {
      state.context += action.payload;
    },
  }
})

export const { addContext } = contextSlice.actions
export default contextSlice.reducer